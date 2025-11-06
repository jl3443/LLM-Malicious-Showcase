import os
from openai import OpenAI

print("=== OpenAI API 连接测试 ===")

# 使用环境变量: export OPENAI_API_KEY=your-key-here
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

print(f"API Key: {api_key[:20]}...{api_key[-20:]}")

try:
    # 创建客户端
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI客户端创建成功")
    
    # 测试简单API调用
    print("正在测试API连接...")
    response = client.chat.completions.create(
        model="gpt-5-nano",
        max_tokens=10,
        messages=[
            {"role": "user", "content": "Hello, just say 'OK'"}
        ]
    )
    
    print("✓ API调用成功!")
    print(f"响应: {response.choices[0].message.content}")
    print(f"模型: {response.model}")
    print(f"使用tokens: {response.usage.total_tokens}")
    
except Exception as e:
    print(f"✗ API调用失败: {type(e).__name__}")
    print(f"错误信息: {e}")
    
    # 检查是否是网络问题
    if "Connection" in str(e):
        print("\n🔍 网络连接问题诊断:")
        print("1. 检查网络连接")
        print("2. 检查防火墙设置")
        print("3. 检查代理设置")
        print("4. 尝试ping openai.com")
    elif "API" in str(e):
        print("\n🔍 API问题诊断:")
        print("1. 检查API Key是否有效")
        print("2. 检查API配额")
        print("3. 检查账户状态")

print("\n=== 测试完成 ===")

