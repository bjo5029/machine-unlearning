# diagnostics.py 
import torch, numpy as np

def print_split_stats(model, loader, device, name):
    """
    주어진 데이터셋(loader)에 대한 모델의 상세 통계(정확도, 손실, 엔트로피, 마진) 출력
    - name: 출력 시 구분을 위한 이름 (예: "Low ES / forget_eval (orig)")
    - loss: 예측 손실
    - entr(entropy): 예측 확률 분포의 불확실성. 높을수록 모델이 헷갈려 함.
    - margin: 가장 높은 확률(top-1)과 두 번째로 높은 확률(top-2)의 차이. 클수록 모델이 확신에 차 있음.
    """
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

# after parts = create_es_partitions_balanced(...)

def _print_partition_set_es(model, dataset_eval, parts, device, batch_size=256):
    """
    생성된 ES 파티션("Low ES", "Medium ES", "High ES")에 속한 샘플들의 실제 평균 ES 점수를 계산하여 출력 (Sanity Check용)
    의도대로 Low -> Medium -> High 순으로 점수가 증가하는지 확인
    """
    import torch, numpy as np, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader
    model = model.to(device).eval()
    extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    # 1) 임베딩/라벨 수집
    loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)
    embs, labs = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            embs.append(extractor(x).squeeze().cpu())
            labs.append(y.cpu())
    E = torch.cat(embs, 0)              # [N,D]
    Y = torch.cat(labs, 0).numpy()      # [N]

    # 2) 클래스 센터와 per-sample ES = d_self - min d_other
    C = int(Y.max()) + 1
    mu = []
    for c in range(C):
        m = (Y==c)
        mu_c = E[m].mean(0) if m.any() else E.mean(0)
        mu.append(mu_c)
    MU = torch.stack(mu,0)              # [C,D]
    dist2 = ((E[:,None,:]-MU[None,:,:])**2).sum(-1)  # [N,C]
    idx = torch.arange(E.size(0))
    d_self = dist2[idx, torch.tensor(Y)]
    dist2[idx, torch.tensor(Y)] = float('inf')
    d_other = dist2.min(1).values
    es = (d_self - d_other).numpy()

    # 3) 파티션별 집합 ES 평균 출력(작을수록 덜 얽힘)
    order = []
    for name in ["Low ES","Medium ES","High ES"]:
        ids = parts[name]
        s = float(es[ids].mean())
        order.append((s, name))
        print(f"[sanity] {name}: set-ES(mean) = {s:.6f}  (n={len(ids)})")
    order_sorted = sorted(order)  # ascending
    print("[sanity] ES scores (ascending):", [f"{n}:{s:.6f}" for s,n in order_sorted])
