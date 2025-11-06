import os
import re
import json
import time
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# ========= Init =========
# 设置Gemini API Key
GEMINI_API_KEY = "AIzaSyBlGfmY1JF32PEv71nsJkgACDRtFMjHjvs"  # 替换为你的API Key
genai.configure(api_key=GEMINI_API_KEY)

# 选择模型
model = genai.GenerativeModel('gemini-1.5-flash')

# ========= IO =========
data_file = './extracted_urls_2000_balanced.csv'
output_file = 'gemini_100_results.json'
reasons_file = 'gemini_100_reasons.json'

df = pd.read_csv(data_file).head(100)
print(f"处理前100个URL...")

if os.path.exists(output_file):
    os.remove(output_file)
    print(f"已删除现有文件: {output_file}")

if os.path.exists(reasons_file):
    os.remove(reasons_file)
    print(f"已删除现有文件: {reasons_file}")

result_dict = {}
result_reasons = {}

# ========= Prompt =========
PROMPT_TEMPLATE = """You are a URL security expert. Evaluate the probability that the given URL is malicious or benign.

Malicious URLs often exhibit abnormal lexical or structural patterns—such as high-entropy or brand-mismatched domains, risky TLDs, suspicious keywords, executable or archive downloads, cloud-storage abuse, redirects/shorteners, and compromised CMS traces—while lacking the consistent branding, clean semantics, and trusted hosting typical of benign sites.

Benign URLs are known reputable sites with clean descriptive paths. Normal encoding alone is neutral.

Examples:
rapreviews.com/archive/2010_04_cityofdirt.html (benign)
Signal: Established review site; semantic and readable path; no suspicious parameters or brand terms. 

http://viva-sport.com/index.php?option=com_virtuemart&page=shop.browse&category_id=19&Itemid=54 (malicious)
Signal: HTTP (no TLS); Joomla/VirtueMart CMS stack with long query string; typical signs of defacement or compromise. 

http://fr57f6yg.gb.net/boa/login.php?... (malicious)
Signal: Subdomain appears high-entropy/random; path contains "boa/login.php" (brand keyword mismatch — brand impersonation); very long session parameters. 

Show me your thinking process, but only end your response with one-line JSON exactly like: {{"score": <float>}} (no extra text).

URL: {url}
Return:"""

# ========= Helpers =========
num_pattern = re.compile(r'(-?\d+(?:\.\d+)?)')

def parse_score_and_reason(text: str) -> tuple[float, str]:
    """Parse final JSON score from the last non-empty line; capture prior lines as reasoning.
    Returns (score, reason_text). Fallback to regex number; else 0.5.
    """
    if text is None:
        return 0.5, ""
    lines = [ln for ln in (text.splitlines()) if ln.strip() != ""]
    reason_text = ""
    json_candidate = None
    if lines:
        json_candidate = lines[-1].strip()
        reason_text = "\n".join(lines[:-1]).strip()
    # Try JSON on the last line
    if json_candidate:
        try:
            obj = json.loads(json_candidate)
            if isinstance(obj, dict) and "score" in obj:
                v = float(obj["score"])
                return max(0.0, min(1.0, v)), reason_text
        except Exception:
            pass
    # Fallback: search last number in full text
    m = num_pattern.search(text)
    if m:
        v = float(m.group(1))
        return max(0.0, min(1.0, v)), reason_text
    return 0.5, reason_text

def call_gemini(url: str, retries: int = 3, backoff: float = 1.0) -> tuple[float, str]:
    """调用Gemini API"""
    last_err = None
    for i in range(retries):
        try:
            prompt = PROMPT_TEMPLATE.format(url=url)
            response = model.generate_content(prompt)
            
            if response.text:
                text = response.text
                score, reason = parse_score_and_reason(text)
                return score, reason
            else:
                print(f"[WARN] Gemini返回空响应: {url}")
                return 0.5, "Empty response from Gemini"
                
        except Exception as e:
            last_err = e
            print(f"[WARN] 第{i+1}次尝试失败: {url}, 错误: {e}")
            if i < retries - 1:  # 不是最后一次尝试
                time.sleep(backoff * (i + 1))
                continue
            else:
                break
    
    print(f"[WARN] Gemini调用失败，使用 0.5: {url}\n  错误: {repr(last_err)}")
    return 0.5, f"Error: {repr(last_err)}"

# ========= Main loop =========
for index, row in tqdm(df.iterrows(), total=len(df)):
    url = row['url']
    true_label = row['type']
    
    print(f"\n处理URL {index+1}/100: {url}")
    print(f"真实标签: {true_label}")

    score, reason = call_gemini(url)
    print(f"Gemini推理: {score}")

    result_dict[url] = score
    result_reasons[url] = reason
    
    print(f"最终概率: {score:.3f}")
    print("-" * 60)

    # 每处理10个URL保存一次
    if len(result_dict) % 10 == 0:
        with open(output_file, 'w') as f:
            json.dump(result_dict, f)
        with open(reasons_file, 'w', encoding='utf-8') as f:
            json.dump(result_reasons, f, ensure_ascii=False, indent=2)
        print(f"\n已保存 {len(result_dict)} 个结果")
    
    # 添加小延迟避免API限流
    time.sleep(0.5)

# ========= Save & Stats =========
with open(output_file, 'w') as f:
    json.dump(result_dict, f, indent=2)
with open(reasons_file, 'w', encoding='utf-8') as f:
    json.dump(result_reasons, f, ensure_ascii=False, indent=2)

print(f"\n完成！总共处理了 {len(result_dict)} 个URL")
print("\n" + "=" * 50)
print("统计信息:")
vals = list(result_dict.values())
print(f"总URL数: {len(result_dict)}")
print(f"平均概率: {sum(vals) / len(vals):.3f}")
print(f"最高概率: {max(vals):.3f}")
print(f"最低概率: {min(vals):.3f}")
