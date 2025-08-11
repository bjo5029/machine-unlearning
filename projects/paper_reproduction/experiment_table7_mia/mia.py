# mia.py 

import numpy as np, torch, torch.nn.functional as F

class BlackBoxBench:
    def __init__(self, s_tr, s_te, t_tr, t_te, k=10):
        """
        MIA 공격을 수행하기 위한 클래스.
        s_tr, s_te: 소스(Source) 분포. 공격 모델을 학습시키기 위한 데이터 (멤버, 비멤버).
                    보통 Retain Set(학습용)과 Test Set을 사용한다.
        t_tr, t_te: 타겟(Target) 분포. 실제 공격을 수행할 대상 데이터 (멤버, 비멤버).
                    보통 Retain Set(평가용)과 Forget Set을 사용한다.
        *_out: 모델의 softmax 출력 (확률 분포)
        *_lab: 실제 라벨
        """
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
        """
        1차원 특징(예: confidence)에 대해 최적의 임계값(threshold)을 찾는다
        이 임계값보다 크면 멤버, 작으면 비멤버로 예측했을 때 가장 정확도가 높은 지점을 찾는다
        """
        vals = np.concatenate((tr, te)); best_acc = 0; best_t = 0
        for v in vals:
            acc = 0.5 * ((tr >= v).mean() + (te < v).mean())
            if acc > best_acc: best_acc, best_t = acc, v
        return best_t

    def _via_feat_src_to_tgt(self, src_tr_feat, src_te_feat, tgt_tr_feat, tgt_te_feat):
        """
        소스 분포(src)에서 클래스별로 최적 임계값을 학습한 뒤, 이 임계값을 타겟 분포(tgt)에 적용하여 MIA 정확도를 계산한다
        소스 라벨: self.s_tr_lab / self.s_te_lab
        타깃 라벨: self.t_tr_lab / self.t_te_lab
        """
        if len(tgt_tr_feat) == 0 or len(tgt_te_feat) == 0:
            return 0.5
        t_mem = t_non = 0
        for c in range(self.k): # 각 클래스(c)에 대해
            # 소스 데이터에서 멤버(tr_c)와 비멤버(te_c)의 특징(feat)을 가져옴
            tr_c = src_tr_feat[self.s_tr_lab == c]
            te_c = src_te_feat[self.s_te_lab == c]
            if len(tr_c) == 0 or len(te_c) == 0:
                continue
            thr = self._thre(tr_c, te_c) # 클래스 c에 대한 최적 임계값(thr) 계산
            # 타겟 데이터의 멤버(mem_c)와 비멤버(non_c)에 이 임계값을 적용하여 맞춘 개수를 셈
            mem_c = tgt_tr_feat[self.t_tr_lab == c]
            non_c = tgt_te_feat[self.t_te_lab == c]
            t_mem += (mem_c >= thr).sum() # 멤버인데 임계값보다 크면 성공
            t_non += (non_c <  thr).sum() # 비멤버인데 임계값보다 작으면 성공
        return 0.5 * (t_mem / len(tgt_tr_feat) + t_non / len(tgt_te_feat)) # 전체 정확도 계산

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
    """데이터 로더의 모든 데이터에 대한 모델의 softmax 출력과 실제 라벨을 수집"""
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
    """MIA 점수 계산 전체 과정을 조율하는 함수"""
    # 1. 소스 분포 데이터 수집
    #    - 소스 멤버(s_tr): retain_train_loader (학습에 사용된 멤버)
    #    - 소스 비멤버(s_te): test_loader (학습에 전혀 사용되지 않은 비멤버)
    s_tr = collect_performance(retain_loader_train, model, device)
    s_te = collect_performance(test_loader,         model, device)

    # 2. 타겟 분포 데이터 수집
    #    - 타겟 멤버(t_tr): retain_loader_test (언러닝 후에도 남아있는 멤버)
    #    - 타겟 비멤버(t_te): forget_loader (언러닝 대상이므로, 성공했다면 비멤버처럼 보여야 함)
    t_tr = collect_performance(retain_loader_test,  model, device)
    t_te = collect_performance(forget_loader,       model, device)

    # 3. BlackBoxBench를 사용하여 confidence 기반의 MIA 점수를 계산
    if any(arr.size == 0 for arr in [s_tr[0], s_te[0], t_tr[0], t_te[0]]):
        return 0.5 # 데이터가 없으면 50% (무작위 추측)
    return BlackBoxBench(s_tr, s_te, t_tr, t_te, 10).run()["confidence"]
