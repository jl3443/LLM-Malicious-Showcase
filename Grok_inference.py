import os
import json
import pandas as pd
from tqdm import tqdm
import requests

# Grok API配置
# 使用环境变量: export XAI_API_KEY=your-key-here
API_KEY = os.environ.get("XAI_API_KEY", "")
if not API_KEY:
    raise ValueError("请设置环境变量 XAI_API_KEY")
API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-4-latest"

# 数据和输出路径
data_file = './sampled_data_2000_balanced.csv'
output_file = 'grok_url_classification_results.json'

# 删除现有结果文件（如果存在）
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"已删除现有文件: {output_file}")

# 读取 CSV 文件
df = pd.read_csv(data_file)

# 创建全新的结果字典
result_dict = {}

print(f"开始重新处理所有 {len(df)} 个URL")

# 循环处理每个 URL
for index, row in tqdm(df.iterrows(), total=len(df)):
    url = row['url']

    # 使用 Grok API
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""Evaluate the probability of this URL being a phishing site.
                    Respond only with a numeric value strictly between 0 (definitely not phishing) and 1 (definitely phishing, malware, defacement or others).
                    Even if uncertain, you must still choose the number closest to your judgment and do NOT output anything else.
                    URL: {url}"""
                }
            ],
            "model": MODEL,
            "stream": False,
            "temperature": 0
        }
        
        response = requests.post(API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content'].strip()
        else:
            response_text = "0.5"

        # 强制转换为float，异常情况默认给0.5
        try:
            probability = float(response_text)
        except:
            probability = 0.5  # 若返回异常则默认中立概率0.5

        probability = min(max(probability, 0), 1)

        result_dict[url] = probability

    except Exception as e:
        print(f"处理URL时出错: {url}, 错误: {e}")
        result_dict[url] = 0.5  # 出错时使用默认值

    # 每处理100个保存一次结果
    if len(result_dict) % 100 == 0:
        with open(output_file, 'w') as f:
            json.dump(result_dict, f)
        print(f"\n已保存 {len(result_dict)} 个结果")

# 最终保存结果
with open(output_file, 'w') as f:
    json.dump(result_dict, f, indent=2)

print(f"\n完成！总共处理了 {len(result_dict)} 个URL")