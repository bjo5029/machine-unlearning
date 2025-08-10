# mia.py (고친 버전의 핵심 부분)

import numpy as np, torch, torch.nn.functional as F

class BlackBoxBench:
    def __init__(self, s_tr, s_te, t_tr, t_te, k=10):
        self.k = k
        self.s_tr_out, self.s_tr_lab = s_tr
        self.s_te_out, self.s_te_lab = s_te
        self.t_tr_out, self.t_tr_lab = t_tr
        self.t_te_out, self.t_te_lab = t_te

        # features
        self.s_tr_conf = self.s_tr_out[np.arange(len(self.s_tr_lab)), self.s_tr_lab]
        self.s_te_conf = self.s_te_out[np.arange(len(self.s_te_lab)), self.s_te_lab]
        self.t_tr_conf = self.t_tr_out[np.arange(len(self.t_tr_lab)), self.t_tr_lab]
        self.t_te_conf = self.t_te_out[np.arange(len(self.t_te_lab)), self.t_te_lab]

        self.s_tr_entr = self._entr(self.s_tr_out); self.s_te_entr = self._entr(self.s_te_out)
        self.t_tr_entr = self._entr(self.t_tr_out); self.t_te_entr = self._entr(self.t_te_out)

        self.s_tr_m_entr = self._m_entr(self.s_tr_out, self.s_tr_lab)
        self.s_te_m_entr = self._m_entr(self.s_te_out, self.s_te_lab)
        self.t_tr_m_entr = self._m_entr(self.t_tr_out, self.t_tr_lab)
        self.t_te_m_entr = self._m_entr(self.t_te_out, self.t_te_lab)

    @staticmethod
    def _log(p, eps=1e-30): return -np.log(np.maximum(p, eps))
    def _entr(self, p): return (p * self._log(p)).sum(1)
    def _m_entr(self, p, l):
        lp = self._log(p); rp = 1 - p; lrp = self._log(rp)
        mp = p.copy(); mp[np.arange(l.size), l] = rp[np.arange(l.size), l]
        mlp = lrp.copy(); mlp[np.arange(l.size), l] = lp[np.arange(l.size), l]
        return (mp * mlp).sum(1)

    def _thre(self, tr, te):
        vals = np.concatenate((tr, te)); best_acc = 0; best_t = 0
        for v in vals:
            acc = 0.5 * ((tr >= v).mean() + (te < v).mean())
            if acc > best_acc: best_acc, best_t = acc, v
        return best_t

    def _via_feat_src_to_tgt(self, src_tr_feat, src_te_feat, tgt_tr_feat, tgt_te_feat):
        """
        소스 분포로 클래스별 threshold 계산 → 타깃 분포에 적용.
        소스 라벨: self.s_tr_lab / self.s_te_lab
        타깃 라벨: self.t_tr_lab / self.t_te_lab
        """
        if len(tgt_tr_feat) == 0 or len(tgt_te_feat) == 0:
            return 0.5
        t_mem = t_non = 0
        for c in range(self.k):
            tr_c = src_tr_feat[self.s_tr_lab == c]
            te_c = src_te_feat[self.s_te_lab == c]
            if len(tr_c) == 0 or len(te_c) == 0:
                continue
            thr = self._thre(tr_c, te_c)
            mem_c = tgt_tr_feat[self.t_tr_lab == c]
            non_c = tgt_te_feat[self.t_te_lab == c]
            t_mem += (mem_c >= thr).sum()
            t_non += (non_c <  thr).sum()
        return 0.5 * (t_mem / len(tgt_tr_feat) + t_non / len(tgt_te_feat))

    def run(self):
        # correctness는 타깃 분포로 계산
        corr_tr = (self.t_tr_out.argmax(1) == self.t_tr_lab).mean()
        corr_te = (self.t_te_out.argmax(1) == self.t_te_lab).mean()
        correctness = 0.5 * (corr_tr + (1 - corr_te))
        return {
            "correctness": correctness,
            "confidence": self._via_feat_src_to_tgt(self.s_tr_conf,  self.s_te_conf,
                                                    self.t_tr_conf,  self.t_te_conf),
            "entropy":    self._via_feat_src_to_tgt(-self.s_tr_entr, -self.s_te_entr,
                                                    -self.t_tr_entr, -self.t_te_entr),
            "m_entropy":  self._via_feat_src_to_tgt(-self.s_tr_m_entr, -self.s_te_m_entr,
                                                    -self.t_tr_m_entr, -self.t_te_m_entr),
        }

def collect_performance(loader, model, device):
    outs, labs = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outs.append(F.softmax(model(x), 1).cpu())
            labs.append(y.cpu())
    if not outs: return np.array([]), np.array([])
    return torch.cat(outs).numpy(), torch.cat(labs).numpy()

def calculate_mia_score(model, retain_loader_train, retain_loader_test, forget_loader, test_loader, device):
    s_tr = collect_performance(retain_loader_train, model, device)  # source train
    s_te = collect_performance(test_loader,         model, device)  # source test
    t_tr = collect_performance(retain_loader_test,  model, device)  # target train
    t_te = collect_performance(forget_loader,       model, device)  # target test (forget)
    if any(arr.size == 0 for arr in [s_tr[0], s_te[0], t_tr[0], t_te[0]]):
        return 0.5
    return BlackBoxBench(s_tr, s_te, t_tr, t_te, 10).run()["confidence"]
