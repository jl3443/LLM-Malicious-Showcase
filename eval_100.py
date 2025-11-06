import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score as f1_sklearn

# 1) 读预测
with open('claudehaiku_100_results.json', 'r') as f:
    raw_preds = json.load(f)

# 2) 读前100真实标签
actual_data = pd.read_csv('./sampled_data_2000_balanced.csv').head(100)

# 3) 映射标签
actual_data['type_numeric'] = actual_data['type'].map(
    {'benign': 0, 'phishing': 1, 'malware': 1, 'defacement': 1}
)

# 4) 按 URL 映射预测，做鲁棒处理
def _safe_to_float(x):
    try:
        v = float(x)
    except Exception:
        v = np.nan
    # clamp 到 [0,1]
    if not np.isnan(v):
        v = max(0.0, min(1.0, v))
    return v

mapped = actual_data['url'].map(raw_preds)               # 可能有 NaN
mapped = mapped.apply(_safe_to_float).fillna(0.5)        # 用 0.5 补缺
y_pred_prob = mapped.values.astype(float)

# 覆盖率检查
coverage = np.isfinite(y_pred_prob).mean()
missing_urls = actual_data.loc[~np.isfinite(mapped), 'url'].tolist()
if missing_urls:
    print(f"[WARN] Missing predictions: {len(missing_urls)} (showing first 5)")
    for u in missing_urls[:5]:
        print("  -", u)

# 5) 真实标签
y_true = actual_data['type_numeric'].values.astype(int)

# 6) ROC / AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Claude Haiku CoT ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Claude Haiku CoT ROC Curve (100 URLs)')
plt.legend(loc='lower right')
plt.show()

# 7) 建议阈值（不改变你主评估的0.2，仅打印建议）
# Youden's J
j_scores = tpr - fpr
best_j_idx = int(np.argmax(j_scores))
best_thresh_j = thresholds[best_j_idx]

# F1最优
f1_scores = []
for th in thresholds:
    y_bin = (y_pred_prob >= th).astype(int)
    f1_scores.append(f1_sklearn(y_true, y_bin))
best_thresh_f1 = thresholds[int(np.argmax(f1_scores))]

print(f"[INFO] Suggested threshold (Youden's J): {best_thresh_j:.3f}")
print(f"[INFO] Suggested threshold (Max F1):     {best_thresh_f1:.3f}")

# 8) 你的固定阈值
TH = 0.2
y_pred = (y_pred_prob >= TH).astype(int)

# 9) 混淆矩阵 + 指标
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign (0)', 'Phish/Mal/Deface (1)'])
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Claude Haiku CoT Confusion Matrix (100 URLs, TH={TH})')
plt.show()

TN, FP, FN, TP = cm.ravel()
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1_score  = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print("=== Claude Haiku CoT Results (100 URLs) ===")
print("Coverage:", f"{coverage*100:.1f}%")
print("True Positives (TP):", TP)
print("False Positives (FP):", FP)
print("True Negatives (TN):", TN)
print("False Negatives (FN):", FN)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("AUROC:", roc_auc)

# 10) 明细（与上面使用同一份分数）
print("\n" + "=" * 50)
print("详细结果:")
for i, (url, true_label, p) in enumerate(zip(actual_data['url'], actual_data['type'], y_pred_prob), start=1):
    pred_label = "Malicious" if p >= TH else "Benign"
    correct = "✓" if (true_label == 'benign' and p < TH) or (true_label != 'benign' and p >= TH) else "✗"
    print(f"{i:2d}. {url[:50]:<50} | True: {true_label:>10} | Pred: {p:.3f} ({pred_label:>9}) | {correct}")
