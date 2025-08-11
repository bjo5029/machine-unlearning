import numpy as np, torch, torch.nn.functional as F

@torch.no_grad()
def calculate_prediction_diff(model_a, model_b, loader, device):
    model_a.eval(); model_b.eval()
    total = diff = 0
    for x, _ in loader:
        x = x.to(device)
        pa = model_a(x).argmax(1)
        pb = model_b(x).argmax(1)
        diff += (pa != pb).sum().item()
        total += x.size(0)
    return 100.0 * diff / total

# ------- MIA (black-box, confidence) -------
class BlackBoxBench:
    def __init__(self, s_tr, s_te, t_tr, t_te, k=10):
        self.k = k
        self.s_tr_out, self.s_tr_lab = s_tr
        self.s_te_out, self.s_te_lab = s_te
        self.t_tr_out, self.t_tr_lab = t_tr
        self.t_te_out, self.t_te_lab = t_te
        self.s_tr_conf = self.s_tr_out[np.arange(len(self.s_tr_lab)), self.s_tr_lab]
        self.s_te_conf = self.s_te_out[np.arange(len(self.s_te_lab)), self.s_te_lab]
        self.t_tr_conf = self.t_tr_out[np.arange(len(self.t_tr_lab)), self.t_tr_lab]
        self.t_te_conf = self.t_te_out[np.arange(len(self.t_te_lab)), self.t_te_lab]

    def _thre(self, tr, te):
        vals = np.concatenate((tr, te)); best_acc = 0; best_t = 0
        for v in vals:
            acc = 0.5 * ((tr >= v).mean() + (te < v).mean())
            if acc > best_acc: best_acc, best_t = acc, v
        return best_t

    def _via_feat_src_to_tgt(self, src_tr_feat, src_te_feat, tgt_tr_feat, tgt_te_feat):
        if len(tgt_tr_feat) == 0 or len(tgt_te_feat) == 0: return 0.5
        t_mem = t_non = 0
        for c in range(self.k):
            tr_c = src_tr_feat[self.s_tr_lab == c]
            te_c = src_te_feat[self.s_te_lab == c]
            if len(tr_c) == 0 or len(te_c) == 0: continue
            thr = self._thre(tr_c, te_c)
            mem_c = tgt_tr_feat[self.t_tr_lab == c]
            non_c = tgt_te_feat[self.t_te_lab == c]
            t_mem += (mem_c >= thr).sum()
            t_non += (non_c <  thr).sum()
        return 0.5 * (t_mem / len(tgt_tr_feat) + t_non / len(tgt_te_feat))

    def run(self):
        return {"confidence": self._via_feat_src_to_tgt(self.s_tr_conf, self.s_te_conf,
                                                        self.t_tr_conf, self.t_te_conf)}

@torch.no_grad()
def _collect_probs_labels(loader, model, device):
    outs, labs = [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        outs.append(F.softmax(model(x), 1).cpu())
        labs.append(y.cpu())
    if not outs: return np.array([]), np.array([])
    return torch.cat(outs).numpy(), torch.cat(labs).numpy()

def calculate_mia_score(model, retain_loader_train, retain_loader_eval, forget_loader_eval, test_loader_eval, device):
    s_tr = _collect_probs_labels(retain_loader_train, model, device)  # src train
    s_te = _collect_probs_labels(test_loader_eval,   model, device)  # src test
    t_tr = _collect_probs_labels(retain_loader_eval, model, device)  # tgt train
    t_te = _collect_probs_labels(forget_loader_eval, model, device)  # tgt test
    if any(arr.size == 0 for arr in [s_tr[0], s_te[0], t_tr[0], t_te[0]]): return 0.5
    return BlackBoxBench(s_tr, s_te, t_tr, t_te, 10).run()["confidence"]
