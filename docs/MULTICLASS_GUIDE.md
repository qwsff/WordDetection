# 多分类违规内容检测详细指南

## 🎯 功能概述

多分类模式是WordDetection系统的高级功能，支持将违规内容细分为5个具体类别，提供更精确的内容审核能力。

### 支持的分类类别

| 类别 | 说明 | 典型示例 |
|------|------|----------|
| **normal** | 正常内容 | "这是一篇很好的文章" |
| **threat** | 威胁内容 | "我要杀了你"、"小心你的家人" |
| **porn** | 色情内容 | "色情相关内容"、"成人影片" |
| **fraud** | 欺诈内容 | "刷单返利"、"投资理财骗局" |
| **insult** | 辱骂内容 | "你这个蠢货"、"垃圾东西" |

## 🚀 快速入门

### 1. 准备多分类训练数据

创建 `data/multiclass_data.csv` 文件：

```csv
text,category
这是一条正常的评论内容,normal
我要杀了你,threat
色情相关内容,porn
刷单返利信息,fraud
你这个蠢货,insult
文章写得很好,normal
死亡威胁内容,threat
成人影片推荐,porn
虚假投资项目,fraud
恶意辱骂他人,insult
```

**数据准备建议**：
- 每个类别至少准备50条高质量样本
- 包含边界案例和容易混淆的内容
- 确保数据平衡，避免某个类别样本过少

### 2. 训练多分类模型

```bash
# 使用统一脚本训练
python run.py train --data-path data/multiclass_data.csv

# 或者直接调用多分类训练器
python -m src.training.multiclass_trainer --data-path data/multiclass_data.csv
```

训练成功后会生成：
- `models/classifiers/multiclass_classifier.pkl` - 多分类器模型
- `models/classifiers/label_encoder.pkl` - 标签编码器

### 3. 启动多分类API服务

```bash
# 方式1：使用统一启动脚本（推荐）
python run.py api --mode multiclass --port 8001

# 方式2：直接运行多分类API
python -m src.api.multiclass_api

# 方式3：使用uvicorn
uvicorn src.api.multiclass_api:app --host 0.0.0.0 --port 8001
```

## 📊 API接口详解

### 1. 单文本检测接口

**POST** `/detect`

**请求参数**：
```json
{
  "text": "我要杀了你",
  "return_probabilities": true,
  "return_embedding": false
}
```

**响应示例**：
```json
{
  "success": true,
  "data": {
    "text": "我要杀了你",
    "predicted_category": "threat",
    "confidence": 0.95,
    "probabilities": {
      "threat": 0.95,
      "insult": 0.03,
      "fraud": 0.01,
      "porn": 0.005,
      "normal": 0.005
    },
    "is_violation": true
  },
  "message": "多分类检测完成"
}
```

### 2. 批量检测接口

**POST** `/batch_detect`

**请求参数**：
```json
{
  "texts": [
    "这是一条正常评论",
    "我要杀了你",
    "色情内容"
  ],
  "batch_size": 32
}
```

**响应示例**：
```json
{
  "success": true,
  "data": [
    {
      "text": "这是一条正常评论",
      "predicted_category": "normal",
      "confidence": 0.92,
      "probabilities": {
        "normal": 0.92,
        "threat": 0.03,
        "insult": 0.02,
        "fraud": 0.02,
        "porn": 0.01
      },
      "is_violation": false
    },
    {
      "text": "我要杀了你",
      "predicted_category": "threat",
      "confidence": 0.95,
      "probabilities": {
        "threat": 0.95,
        "insult": 0.03,
        "fraud": 0.01,
        "porn": 0.005,
        "normal": 0.005
      },
      "is_violation": true
    }
  ],
  "message": "批量检测完成"
}
```

### 3. 分类信息接口

**GET** `/categories`

**响应示例**：
```json
{
  "categories": [
    "fraud",
    "insult", 
    "normal",
    "porn",
    "threat"
  ],
  "count": 5
}
```

### 4. 健康检查接口

**GET** `/health`

**响应示例**：
```json
{
  "status": "healthy",
  "detector_loaded": true,
  "model_info": {
    "embedding_model": "bge-small-zh-v1.5",
    "classifier_type": "LogisticRegression",
    "feature_dimension": 512
  },
  "supported_categories": [
    "fraud",
    "insult",
    "normal", 
    "porn",
    "threat"
  ]
}
```

## 🛠️ 高级配置

### 自定义分类阈值

在配置文件中调整阈值：

```json
{
  "detection": {
    "threshold": 0.7,
    "return_probability": true,
    "return_embedding": false
  }
}
```

### 不同类别的处理策略

可以根据业务需求为不同类别设置不同的处理策略：

```python
# 示例：根据不同类别采取不同措施
category_actions = {
    'normal': 'allow',
    'threat': 'immediate_block',
    'porn': 'review_required', 
    'fraud': 'flag_for_review',
    'insult': 'warning_or_block'
}
```

## 📈 性能优化

### 1. 批量处理优化

```python
# 推荐的批量大小
batch_sizes = {
    'small_batch': 16,    # 小批量，低延迟
    'medium_batch': 32,   # 中等批量，平衡性能
    'large_batch': 64     # 大批量，高吞吐
}
```

### 2. 内存管理

```python
# 清理长时间运行中的内存
import gc

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 3. 并发处理

```python
# 使用异步处理提高并发性能
import asyncio
import aiohttp

async def async_batch_detect(texts, session):
    tasks = []
    for text in texts:
        task = asyncio.create_task(detect_single_async(text, session))
        tasks.append(task)
    return await asyncio.gather(*tasks)
```

## 🔍 质量评估

### 1. 模型评估指标

训练完成后关注以下指标：

```python
# 分类报告示例
"""
              precision    recall  f1-score   support

       fraud       0.92      0.89      0.90        45
      insult       0.88      0.91      0.89        52
      normal       0.94      0.96      0.95       120
        porn       0.85      0.82      0.83        38
      threat       0.90      0.88      0.89        41

    accuracy                           0.91       296
   macro avg       0.89      0.89      0.89       296
weighted avg       0.91      0.91      0.91       296
"""
```

### 2. 持续优化建议

1. **定期更新训练数据**：每季度更新一次训练数据
2. **监控误判案例**：建立误判反馈机制
3. **A/B测试**：对比不同模型版本的效果
4. **用户反馈整合**：将人工审核结果反馈到训练数据中

## 🚨 故障排除

### 常见问题及解决方案

#### 1. 模型加载失败
```bash
# 检查点
- 确认模型文件路径正确
- 验证bge-small-zh-v1.5文件夹完整性
- 检查权限设置
- 尝试重新下载模型
```

#### 2. 分类准确率低
```bash
# 优化方案
- 增加训练数据量
- 平衡各类别样本数量
- 添加更多边界案例
- 调整分类阈值
- 考虑使用更复杂的模型
```

#### 3. API响应慢
```bash
# 性能优化
- 增加批处理大小
- 使用GPU加速（如果可用）
- 优化模型推理代码
- 考虑模型量化
```

#### 4. 内存不足
```bash
# 内存优化
- 减少批处理大小
- 及时清理缓存
- 使用模型量化
- 考虑分布式部署
```

## 📚 最佳实践

### 1. 数据准备最佳实践

```python
# 数据质量检查函数
def validate_training_data(df):
    # 检查必需列
    required_columns = ['text', 'category']
    assert all(col in df.columns for col in required_columns)
    
    # 检查类别平衡性
    category_counts = df['category'].value_counts()
    print("类别分布:")
    print(category_counts)
    
    # 检查文本长度
    df['text_length'] = df['text'].str.len()
    print(f"平均文本长度: {df['text_length'].mean():.1f}")
    
    return True
```

### 2. 生产环境部署建议

```yaml
# docker-compose.yml 示例
version: '3.8'
services:
  word-detection-multiclass:
    build: .
    ports:
      - "8001:8001"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - API_PORT=8001
      - MODEL_PATH=/app/models/bge-small-zh-v1.5
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### 3. 监控告警设置

```python
# 关键指标监控
monitoring_metrics = {
    'api_response_time': {'threshold': 500, 'unit': 'ms'},
    'detection_accuracy': {'threshold': 0.85, 'unit': 'ratio'},
    'model_load_status': {'expected': True, 'check_interval': 60},
    'memory_usage': {'threshold': 80, 'unit': 'percent'}
}
```

## 🎓 学习资源

### 相关文档
- [主项目README](../README.md)
- [配置管理指南](CONFIGURATION_GUIDE.md)
- [API使用说明](USAGE.md)

### 技术参考
- [BGE模型文档](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [FastAPI官方文档](https://fastapi.tiangolo.com/)
- [scikit-learn分类器文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

通过本指南，您应该能够成功部署和使用多分类违规内容检测功能。如有问题，请参考故障排除部分或提交issue。