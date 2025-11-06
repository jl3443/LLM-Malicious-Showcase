#!/usr/bin/env python3
import os
from openai import OpenAI

print("=== OpenAI 可用模型列表 ===")

# 使用环境变量: export OPENAI_API_KEY=your-key-here
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

print(f"API Key: {api_key[:20]}...{api_key[-20:]}")

try:
    # 创建客户端
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI客户端创建成功")
    
    # 获取模型列表
    print("正在获取可用模型列表...")
    models = client.models.list()
    
    print(f"✓ 成功获取模型列表，共 {len(models.data)} 个模型")
    print("\n=== 所有可用模型 ===")
    
    # 按类型分组显示
    gpt_models = []
    other_models = []
    
    for model in models.data:
        if "gpt" in model.id.lower():
            gpt_models.append(model.id)
        else:
            other_models.append(model.id)
    
    print("🤖 GPT模型:")
    for model_id in sorted(gpt_models):
        # 检查是否是gpt-5-nano
        if "gpt-5-nano" in model_id:
            print(f"  ✅ {model_id} (这是你要的模型!)")
        else:
            print(f"  📝 {model_id}")
    
    print("\n🔧 其他模型:")
    for model_id in sorted(other_models):
        print(f"  ⚙️  {model_id}")
    
    # 特别检查gpt-5-nano
    print("\n=== 特别检查 ===")
    if any("gpt-5-nano" in model.id for model in models.data):
        print("✅ gpt-5-nano 模型可用!")
    else:
        print("❌ gpt-5-nano 模型不可用")
        print("建议使用以下替代模型:")
        gpt4_models = [m for m in gpt_models if "gpt-4" in m]
        if gpt4_models:
            print(f"  - {gpt4_models[0]} (推荐)")
    
except Exception as e:
    print(f"✗ 获取模型列表失败: {type(e).__name__}")
    print(f"错误信息: {e}")
    
    if "Connection" in str(e):
        print("\n🔍 网络连接问题，无法访问OpenAI API")
        print("请检查网络设置或代理配置")

print("\n=== 检查完成 ===")