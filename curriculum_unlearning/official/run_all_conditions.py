# run_all_conditions.py (최종 단순화 버전)
import os
import time
import importlib
import argparse
import copy

from run_experiment import run_experiment, load_or_train_original, original_cache_path
from data_es import (
    _embed_all, load_cifar10_with_train_eval, define_forget_set,
    split_retain_forget, _get_sorted_indices, _partition_indices
)

def main(args):
    print(f"Loading configuration from: {args.config_module}.py")
    config_module = importlib.import_module(args.config_module)
    run_config_base = config_module.CONFIG
    
    # --- 사전 계산 단계 ---
    device = run_config_base['device']; bs = run_config_base['batch_size']
    train_ds, train_eval_ds, test_ds = load_cifar10_with_train_eval(run_config_base["data_root"])
    orig_model = load_or_train_original(train_ds, bs, device, run_config_base)
    
    methods_to_run = run_config_base.get("methods_to_run_on_this_gpu")
    print(f"Target partitioning methods for this run: {methods_to_run}")

    global_mu = None
    if 'es' in methods_to_run:
        print("\n[INFO] Pre-calculating global center for ES... (Once)")
        t0 = time.time(); all_embs = _embed_all(orig_model, train_eval_ds, device, bs); global_mu = all_embs.mean(0)
        print(f"Global center pre-calculated in {time.time() - t0:.2f}s")

    print("\n[INFO] Pre-defining 'class' and 'random' forget sets... (Once each)")
    forget_indices_cache = {
        'class': define_forget_set(train_ds, {'forget_set_definition': 'class', **run_config_base}),
        'random': define_forget_set(train_ds, {'forget_set_definition': 'random', **run_config_base})
    }
    print("Forget sets pre-defined.")

    print("\n[INFO] Pre-scoring and sorting all data combinations... (Once per method for this GPU)")
    sorted_indices_cache = {}
    definitions_to_score = ['class', 'random']
    all_possible_methods = ["memorization", "es", "c_proxy", "random"]
    for definition in definitions_to_score:
        total_forget_indices = forget_indices_cache[definition]
        initial_retain_idx, _ = split_retain_forget(len(train_ds), total_forget_indices)
        for method in all_possible_methods:
            if method not in methods_to_run: continue # 이 GPU가 담당하는 메소드만 점수 계산
            sorted_indices_cache[(definition, method, 'forget')] = _get_sorted_indices(total_forget_indices, train_eval_ds, orig_model, run_config_base, method, global_mu)
            sorted_indices_cache[(definition, method, 'retain')] = _get_sorted_indices(initial_retain_idx, train_eval_ds, orig_model, run_config_base, method, global_mu)
    print("All necessary scores pre-calculated and sorted.")
    
    # --- 실험 조합 생성 ---
    all_experiments = []
    definitions = ['class', 'random']
    granularities = ['stage', 'batch', 'sample']
    score_methods = ['memorization', 'es', 'c_proxy']
    pairing_orders = [('easy_first', 'easy_first'), ('easy_first', 'hard_first'), ('hard_first', 'easy_first'), ('hard_first', 'hard_first')]
    
    for definition in definitions:
        for method in methods_to_run:
            base_exp = {'forget_set_definition': definition, 'forget_partitioning_method': method}
            if method == 'random':
                all_experiments.append({'retain_partitioning_method': 'random', 'unlearning_granularity': 'stage', 'use_retain_ordering': True, **base_exp})
            elif method in score_methods:
                for granularity in granularities:
                    for f_order, r_order in pairing_orders:
                        exp = {'retain_partitioning_method': method, 'unlearning_granularity': granularity, 'use_retain_ordering': True, 'forget_partition_ordering': f_order, 'retain_partition_ordering': r_order, **base_exp}
                        all_experiments.append(exp)

    total_conditions = len(all_experiments)
    print("="*60); print(f"Total experiments to run in this process: {total_conditions}"); print("="*60)
    
    start_time = time.time()

    # --- 실험 실행 루프 ---
    for i, exp_params in enumerate(all_experiments, 1):
        run_config = copy.deepcopy(run_config_base); run_config.update(exp_params)
        run_config['rewind_pth'] = original_cache_path(run_config['model_save_dir'], run_config)
        
        f_def = run_config['forget_set_definition']
        f_met = run_config['forget_partitioning_method']
        f_ord = run_config.get('forget_partition_ordering', 'N/A')
        f_gran = run_config['unlearning_granularity']
        
        sorted_f_indices = sorted_indices_cache[(f_def, f_met, 'forget')]
        run_config['precomputed_forget_partitions'] = _partition_indices(sorted_f_indices, f_ord, f_gran)
        
        r_met = run_config['retain_partitioning_method']
        r_ord = run_config.get('retain_partition_ordering', 'N/A')
        sorted_r_indices = sorted_indices_cache[(f_def, r_met, 'retain')]
        run_config['precomputed_retain_partitions'] = _partition_indices(sorted_r_indices, r_ord, f_gran)
            
        fname_parts = [exp_params['forget_set_definition'], exp_params['forget_partitioning_method'], exp_params['unlearning_granularity']]
        if exp_params['forget_partitioning_method'] != 'random':
            fname_parts.append(exp_params.get('forget_partition_ordering','N/A'))
        fname_parts.append("paired")
        if exp_params['forget_partitioning_method'] != 'random':
             fname_parts.append(exp_params.get('retain_partition_ordering', 'N/A'))
        results_filename = f"results_{'_'.join(fname_parts)}.csv"; run_config["results_csv_filename"] = results_filename
        
        print(f"\n--- [{i}/{total_conditions}] Running Experiment ---")
        for key, val in exp_params.items():
            print(f"  - {key}: {val}")
        
        try:
            run_experiment(run_config, train_ds, train_eval_ds, test_ds, orig_model)
        except Exception as e:
            print(f"\n--- [{i}/{total_conditions}] Experiment FAILED ---")
            import traceback; traceback.print_exc()
        finally:
             print(f"\n--- [{i}/{total_conditions}] Experiment FINISHED ---")
             print("="*60)

    end_time = time.time()
    print(f"All experiments in this process completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument('--config_module', type=str, default='config', help='Configuration file to use (e.g., config_gpu0)'); args = parser.parse_args()
    main(args)
    