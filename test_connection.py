import os
import sys
from openai import OpenAI

def test_openai_connection():
    """测试OpenAI API连接"""
    try:
        # 从环境变量获取API key
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("❌ 错误: 请设置环境变量 OPENAI_API_KEY")
            print("   在Linux/Mac: export OPENAI_API_KEY=sk-xxxx")
            print("   在Windows: set OPENAI_API_KEY=sk-xxxx")
            return
        
        print(f"API Key长度: {len(api_key)}")
        print(f"API Key前缀: {api_key[:20]}...")
        
        # 创建客户端
        client = OpenAI(api_key=api_key)
        print("✓ OpenAI客户端创建成功")
        
        # 测试简单API调用
        print("正在测试API调用...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        print("✓ API调用成功!")
        print(f"响应: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ 连接失败: {type(e).__name__}: {e}")
        print(f"错误详情: {str(e)}")
        
        # 提供一些常见问题的解决方案
        if "Connection error" in str(e):
            print("\n🔧 可能的解决方案:")
            print("1. 检查网络连接")
            print("2. 如果在中国大陆，可能需要使用代理")
            print("3. 检查防火墙设置")
        elif "authentication" in str(e).lower():
            print("\n🔧 可能的解决方案:")
            print("1. 检查API key是否正确")
            print("2. 检查API key是否已过期")
            print("3. 检查账户余额")

if __name__ == "__main__":
    test_openai_connection()
