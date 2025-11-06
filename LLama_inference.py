import os
import json
import pandas as pd
from tqdm import tqdm
from llama_api_client import LlamaAPIClient

# 初始化 Llama 客户端 - 直接设置API密钥
client = LlamaAPIClient(
    api_key="LLM|702133619060981|n4zdP2PLq4VUKP0569xE8y8bbv8"
)

# 设置模型
model = "Llama-4-Maverick-17B-128E-Instruct-FP8"

# 数据和输出路径
data_file = './sampled_data_2000_balanced.csv'
output_file = 'llama_url_classification_results.json'

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

    # 使用 Llama API
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Evaluate the probability of this URL being a phishing site.
                    Respond only with a numeric value strictly between 0 (definitely not phishing) and 1 (definitely phishing, malware, defacement or others).
                    Even if uncertain, you must still choose the number closest to your judgment and do NOT output anything else.
                    URL: {url}"""
                }
            ]
        )

        response_text = completion.completion_message.content.text.strip()

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