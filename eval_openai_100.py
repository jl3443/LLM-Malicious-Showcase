import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

print("=== 开始执行OpenAI评估脚本 ===")

# 加载OpenAI预测数据
print("步骤1: 正在加载OpenAI预测数据...")
try:
    with open('openai_100_results.json', 'r') as f:
        openai_predictions = json.load(f)
    print(f"✓ 成功加载OpenAI预测数据，包含 {len(openai_predictions)} 个URL的预测结果")
except Exception as e:
    print(f"✗ 加载OpenAI预测数据失败: {e}")
    exit(1)

# 加载原始实际标签数据
print("步骤2: 正在加载原始实际标签数据...")
try:
    full_data = pd.read_csv('./extracted_urls_2000_balanced.csv')
    print(f"✓ 成功加载完整数据，共 {len(full_data)} 行")
    
    # 检查数据分布
    print(f"完整数据分布: {full_data['type'].value_counts().to_dict()}")
    
    # 找到OpenAI预测数据中URL在原始数据中的位置
    available_urls = list(openai_predictions.keys())
    print(f"OpenAI预测数据中可用的URL数量: {len(available_urls)}")
    
    # 找到这些URL在原始数据中的行号
    url_positions = []
    for url in available_urls:
        if url in full_data['url'].values:
            pos = full_data[full_data['url'] == url].index[0]
            url_positions.append(pos)
    
    if url_positions:
        # 找到包含这些URL的最小范围
        min_pos = min(url_positions)
        max_pos = max(url_positions)
        
        # 取包含这些URL的连续100行，确保包含恶意样本
        start_pos = max(0, min_pos)
        end_pos = min(len(full_data), start_pos + 100)
        
        # 如果范围太小，调整起始位置
        if end_pos - start_pos < 100:
            start_pos = max(0, end_pos - 100)
        
        actual_data = full_data.iloc[start_pos:end_pos].reset_index(drop=True)
        print(f"✓ 找到URL位置范围: {min_pos} - {max_pos}")
        print(f"✓ 取数据范围: {start_pos} - {end_pos}")
        print(f"✓ 实际取到 {len(actual_data)} 行数据")
    else:
        print("⚠️  警告: 在原始数据中找不到OpenAI预测的URL，使用前100行")
        actual_data = full_data.head(100)
    
    # 检查数据分布
    print(f"取到的数据分布: {actual_data['type'].value_counts().to_dict()}")
    
except Exception as e:
    print(f"✗ 加载实际标签数据失败: {e}")
    exit(1)

# 将 phishing, malware, defacement, mal 均标记为1，benign标记为0
print("步骤3: 正在转换标签格式...")
actual_data['type_numeric'] = actual_data['type'].map(
    {'benign': 0, 'phishing': 1, 'malware': 1, 'defacement': 1, 'mal': 1}
)
print(f"✓ 标签转换完成，标签分布: {actual_data['type_numeric'].value_counts().to_dict()}")

# 提取真实标签和OpenAI预测概率
print("步骤4: 正在提取真实标签和OpenAI预测概率...")
y_true = actual_data['type_numeric'].values
print(f"✓ 真实标签提取完成，形状: {y_true.shape}")

# 确保所有URL都有对应的预测值
y_pred_prob_openai = []
missing_urls = []
for url in actual_data['url']:
    if url in openai_predictions:
        y_pred_prob_openai.append(float(openai_predictions[url]))
    else:
        missing_urls.append(url)
        y_pred_prob_openai.append(0.5)  # 默认值

if missing_urls:
    print(f"⚠️  警告: 发现 {len(missing_urls)} 个URL在OpenAI预测中缺失")
    print(f"缺失的URL示例: {missing_urls[:3]}")

y_pred_prob_openai = np.array(y_pred_prob_openai)
print(f"✓ OpenAI预测概率提取完成，形状: {y_pred_prob_openai.shape}")

# 检查是否有NaN值
nan_count = pd.isna(y_pred_prob_openai).sum()
if nan_count > 0:
    print(f"⚠️  警告: 发现 {nan_count} 个NaN值，这可能导致ROC曲线计算失败")
    print("正在处理NaN值...")
    y_pred_prob_openai = pd.Series(y_pred_prob_openai).fillna(0.5).values
    print("✓ NaN值已用0.5填充")
else:
    print("✓ 预测概率数据完整，无NaN值")

# 检查数据是否包含正负样本
positive_samples = np.sum(y_true == 1)
negative_samples = np.sum(y_true == 0)
print(f"正样本(恶意): {positive_samples}, 负样本(良性): {negative_samples}")

if positive_samples == 0 or negative_samples == 0:
    print("✗ 错误: 数据中缺少正样本或负样本，无法计算ROC曲线")
    print("请检查数据分布或增加采样数量")
    exit(1)

# ROC 曲线
print("步骤5: 正在计算ROC曲线...")
try:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob_openai)
    roc_auc = auc(fpr, tpr)
    print(f"✓ ROC曲线计算成功，AUC = {roc_auc:.3f}")
except Exception as e:
    print(f"✗ ROC曲线计算失败: {e}")
    exit(1)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'OpenAI ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('OpenAI ROC Curve (100 URLs)')
plt.legend(loc='lower right')
plt.show()
print("✓ ROC曲线图表显示完成")

# 预测标签（阈值设为0.2）
print("步骤6: 正在计算预测标签...")
y_pred = (y_pred_prob_openai >= 0.2).astype(int)
print(f"✓ 预测标签计算完成，阈值0.2，预测分布: {pd.Series(y_pred).value_counts().to_dict()}")

# Confusion Matrix
print("步骤7: 正在计算混淆矩阵...")
try:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign (0)', 'Phishing/Malware/Defacement (1)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('OpenAI Confusion Matrix (100 URLs)')
    plt.show()
    print("✓ 混淆矩阵计算和显示完成")
except Exception as e:
    print(f"✗ 混淆矩阵计算失败: {e}")
    exit(1)

# TP, FP, TN, FN
print("步骤8: 正在计算性能指标...")
TN, FP, FN, TP = cm.ravel()

print("=== OpenAI Results (100 URLs) ===")
print("True Positives (TP):", TP)
print("False Positives (FP):", FP)
print("True Negatives (TN):", TN)
print("False Negatives (FN):", FN)

# 精确度，召回率和F1分数
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("AUROC:", roc_auc)
print("✓ 性能指标计算完成")

# 显示详细结果
print("\n" + "=" * 50)
print("详细结果:")
print("步骤9: 正在显示详细结果...")
for i, (url, true_label) in enumerate(zip(actual_data['url'], actual_data['type'])):
    pred_prob = openai_predictions.get(url, 0.5)
    pred_label = "Malicious" if pred_prob >= 0.2 else "Benign"
    correct = "✓" if (true_label == 'benign' and pred_prob < 0.2) or (true_label != 'benign' and pred_prob >= 0.2) else "✗"
    print(f"{i+1:2d}. {url[:50]:<50} | True: {true_label:>10} | Pred: {pred_prob:.3f} ({pred_label:>9}) | {correct}")

print("✓ 详细结果显示完成")
print("=== OpenAI评估脚本执行完成 ===") 