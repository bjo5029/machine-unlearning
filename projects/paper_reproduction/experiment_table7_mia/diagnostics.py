# diagnostics.py (run_experiment.py와 같은 폴더)
import torch, numpy as np

def print_split_stats(model, loader, device, name):
    import torch.nn.functional as F
    model.eval(); losses=[]; entrs=[]; margins=[]; labs=[]; preds=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device)
            z = model(x); p = torch.softmax(z,1)
            loss = F.cross_entropy(z, y, reduction='none')
            top2 = torch.topk(z, k=2, dim=1).values
            margin = (top2[:,0]-top2[:,1]).cpu().numpy()
            entr = -(p*torch.log(torch.clamp(p,1e-9,1))).sum(1).cpu().numpy()
            losses.append(loss.cpu().numpy()); entrs.append(entr)
            margins.append(margin); labs.append(y.cpu().numpy()); preds.append(z.argmax(1).cpu().numpy())
    losses=np.concatenate(losses); entrs=np.concatenate(entrs)
    margins=np.concatenate(margins); labs=np.concatenate(labs); preds=np.concatenate(preds)
    acc = (labs==preds).mean()*100
    print(f"[{name}] n={len(labs)}  Acc={acc:.2f}  loss={losses.mean():.3f}  entr={entrs.mean():.3f}  margin={margins.mean():.3f}")
