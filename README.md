# URL Classification Project

使用多种LLM模型进行URL恶意检测和分类的项目。

## 功能

- 使用OpenAI GPT模型进行URL分类
- 支持多种模型：Claude, Gemini, Grok, Llama等
- 批量处理URL并生成分类结果
- 评估模型性能

## 安装

```bash
pip install openai pandas tqdm
```

## 配置

设置OpenAI API Key：

**Linux/Mac:**
```bash
export OPENAI_API_KEY=sk-xxxx
```

**Windows:**
```cmd
set OPENAI_API_KEY=sk-xxxx
```

**PowerShell:**
```powershell
$env:OPENAI_API_KEY="sk-xxxx"
```

## 使用方法

### 运行OpenAI推理
```bash
python inference_100.py
```

### 测试API连接
```bash
python test_connection.py
```

### 评估结果
```bash
python eval_openai_100.py
```

### 生成混淆矩阵可视化
```bash
python generate_confusion_matrix.py
```
这将生成一个美观的混淆矩阵图片 `confusion_matrix.png`，包含详细的性能指标。

## 文件说明

### 核心脚本
- `inference_100.py` - OpenAI GPT模型推理脚本
- `eval_openai_100.py` - 评估脚本
- `test_connection.py` - API连接测试脚本
- `generate_confusion_matrix.py` - 生成美观的混淆矩阵可视化

### 文档
- `PROMPT.md` - 📋 Prompt工程展示文档（展示完整的prompt设计）
- `README.md` - 项目说明文档

### 数据
- `extracted_urls_2000_balanced_shuffled.csv` - 数据集文件

## 注意事项

- 请确保设置了正确的API Key环境变量
- 数据集文件较大，可能需要一些时间处理
- 建议使用虚拟环境运行

## License

MIT

