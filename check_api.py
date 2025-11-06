import os
import time
from openai import OpenAI

def check_api_status():
    """检查OpenAI API状态"""
    print("=" * 50)
    print("检查OpenAI API状态")
    print("=" * 50)
    
    # 检查API Key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        print(f"✅ 环境变量OPENAI_API_KEY已设置")
        print(f"   Key前10位: {api_key[:10]}...")
    else:
        print("❌ 环境变量OPENAI_API_KEY未设置")
    
    # 使用环境变量或硬编码Key（仅用于测试）
    test_key = api_key if api_key else os.environ.get("OPENAI_API_KEY", "")
    if not test_key:
        print("❌ 未找到API Key，请设置环境变量 OPENAI_API_KEY")
        return
    
    print(f"使用Key前10位: {test_key[:10]}...")
    
    # 测试API连接
    client = OpenAI(api_key=test_key)
    
    try:
        print("\n🔄 测试API连接...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_completion_tokens=50,
            temperature=0.0,
            messages=[
                {"role": "user", "content": "Hello, just say 'API working' and nothing else."}
            ]
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"✅ API连接成功!")
        print(f"   响应时间: {response_time:.2f}秒")
        print(f"   模型: {response.model}")
        print(f"   响应内容: {response.choices[0].message.content}")
        
        # 测试模型列表
        print("\n🔄 获取可用模型列表...")
        models = client.models.list()
        available_models = [model.id for model in models.data]
        print(f"✅ 可用模型数量: {len(available_models)}")
        print("前10个模型:")
        for i, model_id in enumerate(available_models[:10]):
            print(f"   {i+1}: {model_id}")
            
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        
        # 检查是否是认证问题
        if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
            print("   🔍 可能是API Key无效或过期")
        elif "rate limit" in str(e).lower():
            print("   🔍 可能是API限流")
        elif "quota" in str(e).lower():
            print("   🔍 可能是API配额用完")
        else:
            print("   🔍 其他错误，请检查网络连接")

if __name__ == "__main__":
    check_api_status()

