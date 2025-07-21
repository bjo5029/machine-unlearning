import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import pandas as pd
from scipy import stats
import time
import copy
import itertools
import os
from sklearn.linear_model import LogisticRegression

# ===================================================================
# 1. 실험 환경 설정
# ===================================================================
CONFIG = {
    "run_training": True,
    "model_save_dir": "saved_models",
    "num_runs": 3,
    "epochs": 30,
    "unlearn_epochs": 10,
    "batch_size": 256,
    "lr": 0.1,
    "unlearn_lr": 0.01,
    "unlearn_lr_neggrad": 1e-4,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "forget_set_size": 3000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "l1_lambda": 1e-5,
    "neggrad_plus_alpha": 0.2,
    "salun_sparsity": 0.5,
    "scrub_alpha": 0.5,
}

print(f"Using device: {CONFIG['device']}")

# ===================================================================
# 2. 모델 및 데이터 관련 헬퍼 함수
# ===================================================================
def get_model():
    """ResNet-18 모델 생성 (CIFAR-10용)"""
    model = models.resnet18(weights=None, num_classes=10)
    return model.to(CONFIG['device'])

def train_model(model, train_loader, epochs, lr, is_unlearning=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
    if not is_unlearning:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if not is_unlearning:
            scheduler.step()
        
        epoch_end_time = time.time()
        print(f"    Epoch {epoch+1}/{epochs} completed in {epoch_end_time - epoch_start_time:.2f}s")

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

# ===================================================================
# 3. ES(Entanglement Score) 기반 데이터 분할
# ===================================================================
def create_es_partitions(original_model, train_dataset):
    print("\nCreating ES partitions...")
    start_time = time.time()
    embedding_extractor = nn.Sequential(*list(original_model.children())[:-1])
    embedding_extractor.eval()
    
    all_embeddings = []
    loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    print("  Extracting embeddings from original model...")
    for i, (inputs, _) in enumerate(loader):
        if (i + 1) % 40 == 0:
            print(f"    Batch {i+1}/{len(loader)}")
        with torch.no_grad():
            inputs = inputs.to(CONFIG['device'])
            embeddings = embedding_extractor(inputs).squeeze()
            all_embeddings.append(embeddings.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)

    centroid = all_embeddings.mean(dim=0)
    distances = torch.sum((all_embeddings - centroid) ** 2, dim=1)
    sorted_indices = torch.argsort(distances, descending=True).numpy()

    fs_size = CONFIG['forget_set_size']
    partitions = {
        "Low ES": sorted_indices[:fs_size],
        "Medium ES": sorted_indices[fs_size : 2 * fs_size],
        "High ES": sorted_indices[2 * fs_size : 3 * fs_size],
    }
    end_time = time.time()
    print(f"ES partitions created in {end_time - start_time:.2f}s.")
    return partitions

# ===================================================================
# 4. 언러닝(Unlearning) 알고리즘 구현
# ===================================================================
class RelabelDataset(Dataset):
    def __init__(self, original_dataset, num_classes=10):
        self.original_dataset = original_dataset
        self.num_classes = num_classes
        self.new_labels = [torch.randint(0, num_classes, (1,)).item() for _ in range(len(original_dataset))]
        
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, original_label = self.original_dataset[idx]
        new_label = self.new_labels[idx]
        while new_label == original_label:
            new_label = torch.randint(0, self.num_classes, (1,)).item()
        return image, new_label

def unlearn_finetune(original_model, retain_loader, config):
    unlearned_model = copy.deepcopy(original_model)
    train_model(unlearned_model, retain_loader, config['unlearn_epochs'], config['unlearn_lr'], is_unlearning=True)
    return unlearned_model

def unlearn_neggrad(original_model, forget_loader, config):
    unlearned_model = copy.deepcopy(original_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=config['unlearn_lr_neggrad'])
    unlearned_model.train()
    for _ in range(config['unlearn_epochs']):
        for inputs, targets in forget_loader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            optimizer.zero_grad()
            outputs = unlearned_model(inputs)
            loss = -criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return unlearned_model

def unlearn_l1_sparse(original_model, retain_loader, config):
    unlearned_model = copy.deepcopy(original_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=config['unlearn_lr'], momentum=config['momentum'])
    unlearned_model.train()
    for _ in range(config['unlearn_epochs']):
        for inputs, targets in retain_loader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            optimizer.zero_grad()
            outputs = unlearned_model(inputs)
            l1_penalty = 0.
            for param in unlearned_model.parameters():
                l1_penalty += torch.abs(param).sum()
            loss = criterion(outputs, targets) + config['l1_lambda'] * l1_penalty
            loss.backward()
            optimizer.step()
    return unlearned_model

def unlearn_neggrad_plus(original_model, retain_loader, forget_loader, config):
    unlearned_model = copy.deepcopy(original_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=config['unlearn_lr'])
    unlearned_model.train()
    for _ in range(config['unlearn_epochs']):
        retain_iter = iter(itertools.cycle(retain_loader))
        for forget_inputs, forget_targets in forget_loader:
            retain_inputs, retain_targets = next(retain_iter)
            retain_inputs, retain_targets = retain_inputs.to(config['device']), retain_targets.to(config['device'])
            forget_inputs, forget_targets = forget_inputs.to(config['device']), forget_targets.to(config['device'])
            optimizer.zero_grad()
            retain_outputs = unlearned_model(retain_inputs)
            loss_retain = criterion(retain_outputs, retain_targets)
            forget_outputs = unlearned_model(forget_inputs)
            loss_forget = -criterion(forget_outputs, forget_targets)
            loss = loss_retain + config['neggrad_plus_alpha'] * loss_forget
            loss.backward()
            optimizer.step()
    return unlearned_model

def unlearn_scrub(original_model, retain_loader, forget_loader, config):
    unlearned_model = copy.deepcopy(original_model)
    teacher_model = copy.deepcopy(original_model)
    teacher_model.eval()
    criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.SGD(unlearned_model.parameters(), lr=config['unlearn_lr'])
    unlearned_model.train()
    for _ in range(config['unlearn_epochs']):
        retain_iter = iter(itertools.cycle(retain_loader))
        for forget_inputs, _ in forget_loader:
            retain_inputs, retain_targets = next(retain_iter)
            retain_inputs, retain_targets = retain_inputs.to(config['device']), retain_targets.to(config['device'])
            forget_inputs = forget_inputs.to(config['device'])
            optimizer.zero_grad()
            retain_outputs = unlearned_model(retain_inputs)
            loss_retain = criterion(retain_outputs, retain_targets)
            student_forget_out = F.log_softmax(unlearned_model(forget_inputs), dim=1)
            teacher_forget_out = F.softmax(teacher_model(forget_inputs), dim=1)
            loss_forget = -kl_criterion(student_forget_out, teacher_forget_out)
            loss = (1 - config['scrub_alpha']) * loss_retain + config['scrub_alpha'] * loss_forget
            loss.backward()
            optimizer.step()
    return unlearned_model

def unlearn_random_label(original_model, forget_set, config):
    unlearned_model = copy.deepcopy(original_model)
    relabel_dataset = RelabelDataset(forget_set)
    relabel_loader = DataLoader(relabel_dataset, batch_size=config['batch_size'], shuffle=True)
    train_model(unlearned_model, relabel_loader, config['unlearn_epochs'], config['unlearn_lr'], is_unlearning=True)
    return unlearned_model

def unlearn_salun(original_model, forget_set, config):
    unlearned_model = copy.deepcopy(original_model)
    saliency = [torch.zeros_like(p) for p in unlearned_model.parameters()]
    criterion = nn.CrossEntropyLoss()
    forget_loader = DataLoader(forget_set, batch_size=config['batch_size'])
    for inputs, targets in forget_loader:
        inputs, targets = inputs.to(config['device']), targets.to(config['device'])
        unlearned_model.zero_grad()
        outputs = unlearned_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        for i, param in enumerate(unlearned_model.parameters()):
            if param.grad is not None:
                saliency[i] += param.grad.abs()
    flat_saliency = torch.cat([s.flatten() for s in saliency])
    k = int(len(flat_saliency) * config['salun_sparsity'])
    threshold, _ = torch.kthvalue(flat_saliency, k)
    masks = [(s > threshold).float() for s in saliency]
    relabel_dataset = RelabelDataset(forget_set)
    relabel_loader = DataLoader(relabel_dataset, batch_size=config['batch_size'], shuffle=True)
    optimizer = optim.SGD(unlearned_model.parameters(), lr=config['unlearn_lr'], momentum=config['momentum'])
    unlearned_model.train()
    for _ in range(config['unlearn_epochs']):
        for inputs, targets in relabel_loader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            optimizer.zero_grad()
            outputs = unlearned_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            for i, param in enumerate(unlearned_model.parameters()):
                if param.grad is not None:
                    param.grad *= masks[i]
            optimizer.step()
    return unlearned_model

# ===================================================================
# 5. MIA(Membership Inference Attack) 구현
# ===================================================================
def get_prediction_outputs(model, loader):
    model.eval()
    outputs_list = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(CONFIG['device'])
            outputs = torch.softmax(model(inputs), dim=1)
            outputs_list.append(outputs.cpu().numpy())
    return np.concatenate(outputs_list, axis=0)

def calculate_mia_score(unlearned_model, retain_eval_loader, forget_loader, test_loader):
    retain_outputs = get_prediction_outputs(unlearned_model, retain_eval_loader)
    test_outputs = get_prediction_outputs(unlearned_model, test_loader)
    train_labels = np.ones(len(retain_outputs))
    test_labels = np.zeros(len(test_outputs))
    attack_X = np.concatenate([retain_outputs, test_outputs])
    attack_y = np.concatenate([train_labels, test_labels])
    attack_model = LogisticRegression(solver='liblinear')
    attack_model.fit(attack_X, attack_y)
    forget_outputs = get_prediction_outputs(unlearned_model, forget_loader)
    mia_predictions = attack_model.predict(forget_outputs)
    mia_score = np.mean(mia_predictions == 0)
    return mia_score

# ===================================================================
# 6. 메인 실험 루프 
# ===================================================================
def main():
    save_dir = CONFIG["model_save_dir"]
    if CONFIG["run_training"]:
        os.makedirs(save_dir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    all_methods = ["Retrain", "Original", "Fine-tune", "L1-sparse", "NegGrad", "NegGrad+", "SCRUB", "SalUn", "Random-label"]
    results = {
        "Forget Acc": {method: {es: [] for es in ["Low ES", "Medium ES", "High ES"]} for method in all_methods},
        "Retain Acc": {method: {es: [] for es in ["Low ES", "Medium ES", "High ES"]} for method in all_methods},
        "Test Acc": {method: {es: [] for es in ["Low ES", "Medium ES", "High ES"]} for method in all_methods},
        "MIA": {method: {es: [] for es in ["Low ES", "Medium ES", "High ES"]} for method in all_methods},
    }

    for run in range(CONFIG['num_runs']):
        print(f"\n{'='*20} Starting Run {run+1}/{CONFIG['num_runs']} {'='*20}")

        original_model = get_model()
        original_model_path = os.path.join(save_dir, f"run_{run}_original_model.pth")
        partitions_path = os.path.join(save_dir, f"run_{run}_es_partitions.pth")

        if CONFIG["run_training"]:
            print("\n[TRAINING MODE] Step 1/5: Training original model...")
            run_start_time = time.time()
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
            train_model(original_model, train_loader, CONFIG['epochs'], CONFIG['lr'])
            torch.save(original_model.state_dict(), original_model_path)
            print(f"Original model trained and saved in {time.time() - run_start_time:.2f}s")
            
            es_partitions = create_es_partitions(original_model, train_dataset)
            torch.save(es_partitions, partitions_path)
        else:
            print("\n[EVALUATION MODE] Loading pre-trained original model and ES partitions...")
            original_model.load_state_dict(torch.load(original_model_path, map_location=CONFIG['device']))
            es_partitions = torch.load(partitions_path)

        for es_level, forget_indices in es_partitions.items():
            print(f"\n--- Processing ES Level: {es_level} ---")
            
            all_indices = np.arange(len(train_dataset))
            retain_indices = np.setdiff1d(all_indices, forget_indices, assume_unique=True)
            
            retain_set = Subset(train_dataset, retain_indices)
            forget_set = Subset(train_dataset, forget_indices)
            
            retain_loader = DataLoader(retain_set, batch_size=CONFIG['batch_size'], shuffle=True)
            retain_eval_loader = DataLoader(retain_set, batch_size=CONFIG['batch_size'], shuffle=False)
            forget_loader = DataLoader(forget_set, batch_size=CONFIG['batch_size'], shuffle=False)

            retrained_model = get_model()
            retrained_model_path = os.path.join(save_dir, f"run_{run}_{es_level.replace(' ', '')}_retrained.pth")

            if CONFIG["run_training"]:
                print(f"\n[TRAINING MODE] Step 2/5: Training retrained model for {es_level}...")
                retrain_start_time = time.time()
                train_model(retrained_model, retain_loader, CONFIG['epochs'], CONFIG['lr'])
                torch.save(retrained_model.state_dict(), retrained_model_path)
                print(f"Retrained model trained and saved in {time.time() - retrain_start_time:.2f}s")
            else:
                print(f"\n[EVALUATION MODE] Loading retrained model for {es_level}...")
                retrained_model.load_state_dict(torch.load(retrained_model_path, map_location=CONFIG['device']))
            
            print("\n[Step 3/5] Evaluating retrained model...")
            retrained_accs = {
                'forget': evaluate_model(retrained_model, forget_loader),
                'retain': evaluate_model(retrained_model, retain_eval_loader),
                'test': evaluate_model(retrained_model, test_loader)
            }
            retrained_mia = calculate_mia_score(retrained_model, retain_eval_loader, forget_loader, test_loader)
            
            print(f"  Retrained Accs -> Forget: {retrained_accs['forget']:.2f}%, Retain: {retrained_accs['retain']:.2f}%, Test: {retrained_accs['test']:.2f}%")
            print(f"  Retrained MIA -> {retrained_mia:.3f}")

            # Store Retrain results for this run
            results["Forget Acc"]["Retrain"][es_level].append(retrained_accs['forget'])
            results["Retain Acc"]["Retrain"][es_level].append(retrained_accs['retain'])
            results["Test Acc"]["Retrain"][es_level].append(retrained_accs['test'])
            results["MIA"]["Retrain"][es_level].append(retrained_mia)

            unlearning_methods = {
                "Original": lambda: copy.deepcopy(original_model),
                "Fine-tune": lambda: unlearn_finetune(original_model, retain_loader, CONFIG),
                "L1-sparse": lambda: unlearn_l1_sparse(original_model, retain_loader, CONFIG),
                "NegGrad": lambda: unlearn_neggrad(original_model, forget_loader, CONFIG),
                "NegGrad+": lambda: unlearn_neggrad_plus(original_model, retain_loader, forget_loader, CONFIG),
                "SCRUB": lambda: unlearn_scrub(original_model, retain_loader, forget_loader, CONFIG),
                "SalUn": lambda: unlearn_salun(original_model, forget_set, CONFIG),
                "Random-label": lambda: unlearn_random_label(original_model, forget_set, CONFIG),
            }
            
            print("\n[Step 4/5] Applying and evaluating unlearning methods...")
            for method_name, method_fn in unlearning_methods.items():
                unlearned_model_path = os.path.join(save_dir, f"run_{run}_{es_level.replace(' ', '')}_{method_name}_unlearned.pth")
                
                if CONFIG["run_training"]:
                    print(f"  > [TRAINING MODE] Applying method: {method_name}...")
                    method_start_time = time.time()
                    unlearned_model = method_fn()
                    torch.save(unlearned_model.state_dict(), unlearned_model_path)
                    print(f"    Method {method_name} applied and saved in {time.time() - method_start_time:.2f}s")
                else:
                    print(f"  > [EVALUATION MODE] Loading unlearned model for method: {method_name}...")
                    unlearned_model = get_model() 
                    unlearned_model.load_state_dict(torch.load(unlearned_model_path, map_location=CONFIG['device']))

                unlearned_accs = {
                    'forget': evaluate_model(unlearned_model, forget_loader),
                    'retain': evaluate_model(unlearned_model, retain_eval_loader),
                    'test': evaluate_model(unlearned_model, test_loader)
                }
                mia_score = calculate_mia_score(unlearned_model, retain_eval_loader, forget_loader, test_loader)

                results["Forget Acc"][method_name][es_level].append(unlearned_accs['forget'])
                results["Retain Acc"][method_name][es_level].append(unlearned_accs['retain'])
                results["Test Acc"][method_name][es_level].append(unlearned_accs['test'])
                results["MIA"][method_name][es_level].append(mia_score)
                
                print(f"    - Unlearned Accs -> Forget: {unlearned_accs['forget']:.2f}%, Retain: {unlearned_accs['retain']:.2f}%, Test: {unlearned_accs['test']:.2f}%")
                print(f"    - MIA Score -> {mia_score:.3f}")

    # ===================================================================
    # 7. 결과 정리 및 출력
    # ===================================================================
    print(f"\n{'='*20} Final Results {'='*20}")
    
    def format_results(data):
        mean = np.mean(data)
        if len(data) < 2: return f"{mean:.3f}"
        sem = stats.sem(data)
        ci = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
        return f"{mean:.3f} ± {ci:.3f}"

    for es_level in ["Low ES", "Medium ES", "High ES"]:
        print(f"\n--- Results for {es_level} ---")
        output_data = []
        for method in all_methods:
            row = {'Method': method}
            row['Forget Acc'] = format_results(results["Forget Acc"][method][es_level])
            row['Retain Acc'] = format_results(results["Retain Acc"][method][es_level])
            row['Test Acc'] = format_results(results["Test Acc"][method][es_level])
            row['MIA'] = format_results(results["MIA"][method][es_level])
            output_data.append(row)
        
        df = pd.DataFrame(output_data)
        print(df.to_string(index=False))

if __name__ == '__main__':
    main()
