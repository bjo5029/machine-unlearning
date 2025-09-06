import torch

def calculate_tow(unlearned_accs, retrained_accs):
    """
    ToW (Tug-of-War)
      - retain/forget/test 각 정확도의 '재학습 모델'과의 차이가 작을수록 높음
      - 세 지표의 (1 - 상대오차)를 곱해 종합 점수화
    """
    def gap(a, b):  # a,b are percentages [0,100]
        return abs(a - b) / 100.0
    la_s = gap(unlearned_accs['forget'], retrained_accs['forget'])
    la_r = gap(unlearned_accs['retain'], retrained_accs['retain'])
    la_t = gap(unlearned_accs['test'],   retrained_accs['test'])
    return (1 - la_s) * (1 - la_r) * (1 - la_t)

@torch.no_grad()
def calculate_prediction_diff(unlearned_model, retrained_model, loader, device):
    """
    전체(또는 학습셋)에서 두 모델의 예측이 다른 비율(%) — 원 논문 표기 맞춤.
    """
    unlearned_model.eval(); retrained_model.eval()
    diff = total = 0
    for x, _ in loader:
        x = x.to(device)
        p1 = unlearned_model(x).argmax(1)
        p2 = retrained_model(x).argmax(1)
        diff  += (p1 != p2).sum().item()
        total += x.size(0)
    return 100.0 * diff / total
