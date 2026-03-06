# 配置管理详细指南

## 📋 配置文件结构

WordDetection系统采用统一的配置管理机制，所有配置参数都集中在 `src/config/config.json` 文件中。

### 主配置文件

```json
{
  "model": {
    "embedding_model_path": "models/bge-small-zh-v1.5",
    "classifier_save_path": "models/classifiers/forbidden_classifier.pkl",
    "multiclass_classifier_path": "models/classifiers/multiclass_classifier.pkl",
    "label_encoder_path": "models/classifiers/label_encoder.pkl"
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": true,
    "workers": 1
  },
  "training": {
    "data_path": "data/labeled_data.csv",
    "test_size": 0.2,
    "random_state": 42,
    "batch_size": 32
  },
  "detection": {
    "threshold": 0.5,
    "return_probability": true,
    "return_embedding": false
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/detection.log"
  }
}
```

## 🛠️ 配置项详解

### 模型配置 (model)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `embedding_model_path` | string | `"models/bge-small-zh-v1.5"` | 嵌入模型路径 |
| `classifier_save_path` | string | `"models/classifiers/forbidden_classifier.pkl"` | 二分类器保存路径 |
| `multiclass_classifier_path` | string | `"models/classifiers/multiclass_classifier.pkl"` | 多分类器保存路径 |
| `label_encoder_path` | string | `"models/classifiers/label_encoder.pkl"` | 标签编码器路径 |

### API配置 (api)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `host` | string | `"0.0.0.0"` | 监听主机地址 |
| `port` | integer | `8000` | 监听端口 |
| `reload` | boolean | `true` | 开发模式热重载 |
| `workers` | integer | `1` | 工作进程数 |

### 训练配置 (training)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `data_path` | string | `"data/labeled_data.csv"` | 训练数据路径 |
| `test_size` | float | `0.2` | 测试集比例 (0-1) |
| `random_state` | integer | `42` | 随机种子 |
| `batch_size` | integer | `32` | 批处理大小 |

### 检测配置 (detection)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `threshold` | float | `0.5` | 分类阈值 (0-1) |
| `return_probability` | boolean | `true` | 是否返回概率 |
| `return_embedding` | boolean | `false` | 是否返回嵌入向量 |

### 日志配置 (logging)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `level` | string | `"INFO"` | 日志级别 |
| `format` | string | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | 日志格式 |
| `file` | string | `"logs/detection.log"` | 日志文件路径 |

## 🔧 配置管理方式

### 1. 文件配置（推荐）

直接编辑 `src/config/config.json` 文件：

```json
{
  "api": {
    "port": 8080,
    "host": "127.0.0.1"
  },
  "training": {
    "data_path": "/custom/path/to/data.csv"
  }
}
```

### 2. 环境变量覆盖

使用环境变量临时覆盖配置：

```bash
# 设置API端口
export API_PORT=8080

# 设置模型路径
export MODEL_EMBEDDING_MODEL_PATH=/path/to/custom/model

# 设置训练数据路径
export TRAINING_DATA_PATH=/path/to/training/data.csv

# 设置日志级别
export LOGGING_LEVEL=DEBUG
```

### 3. 命令行参数

使用启动脚本的命令行参数：

```bash
# 训练时指定数据路径
python run.py train --data-path /path/to/data.csv --model-path /path/to/model

# API启动时指定端口和主机
python run.py api --host 0.0.0.0 --port 8001 --mode multiclass

# 仅使用本地模型
python run.py api --local-model-only
```

### 4. 程序内动态配置

在代码中动态修改配置：

```python
from src.config.manager import config_manager

# 获取配置
api_config = config_manager.get_api_config()
model_config = config_manager.get_model_config()

# 更新配置
config_manager.config.set('api.port', 8080)
config_manager.config.set('training.test_size', 0.3)

# 批量更新
updates = {
    'api.host': '192.168.1.100',
    'api.port': 9000,
    'detection.threshold': 0.7
}
config_manager.config.update(updates)
```

## 🎯 不同场景的配置建议

### 开发环境配置

```json
{
  "api": {
    "host": "127.0.0.1",
    "port": 8000,
    "reload": true,
    "workers": 1
  },
  "logging": {
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  },
  "detection": {
    "return_embedding": true
  }
}
```

### 生产环境配置

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": false,
    "workers": 4
  },
  "logging": {
    "level": "WARNING",
    "file": "/var/log/word-detection/detection.log"
  },
  "detection": {
    "threshold": 0.7,
    "return_probability": false
  }
}
```

### 高性能场景配置

```json
{
  "api": {
    "workers": 8,
    "reload": false
  },
  "training": {
    "batch_size": 64
  },
  "detection": {
    "return_embedding": false
  }
}
```

## 🔍 配置验证

系统提供配置验证功能：

```python
from src.config.manager import config_manager

# 验证配置
errors = config_manager.validate_config()
if errors:
    print("配置验证失败:")
    for error in errors:
        print(f"  - {error}")
else:
    print("配置验证通过")
```

### 常见验证规则

1. **路径验证**：检查文件和目录路径是否存在
2. **数值范围**：验证数字参数是否在合理范围内
3. **必需字段**：检查必需的配置项是否存在
4. **类型检查**：验证配置值的类型是否正确

## 📊 配置监控

### 运行时配置查看

```bash
# 通过API查看当前配置
curl http://localhost:8000/config

# 或在Python中查看
from src.config.manager import config_manager
print(config_manager.config.all_config)
```

### 配置变更监控

```python
# 监控配置变更
def on_config_change(key, old_value, new_value):
    print(f"配置变更: {key} 从 {old_value} 变更为 {new_value}")

config_manager.config.add_listener(on_config_change)
```

## 🚨 故障排除

### 常见配置问题

#### 1. 端口被占用
```bash
# 检查端口占用
netstat -an | grep :8000

# 解决方案：修改端口配置
{
  "api": {
    "port": 8080
  }
}
```

#### 2. 路径配置错误
```bash
# 检查路径是否存在
ls -la /path/to/model

# 解决方案：使用相对路径或正确的绝对路径
{
  "model": {
    "embedding_model_path": "models/bge-small-zh-v1.5"
  }
}
```

#### 3. 权限问题
```bash
# 检查文件权限
ls -la models/classifiers/

# 解决方案：调整权限
chmod 644 models/classifiers/*.pkl
```

#### 4. 内存不足
```json
{
  "api": {
    "workers": 1
  },
  "training": {
    "batch_size": 16
  }
}
```

## 📚 最佳实践

### 1. 配置文件组织

```
configs/
├── development.json      # 开发环境配置
├── production.json       # 生产环境配置
├── staging.json          # 测试环境配置
└── local.json           # 本地开发配置
```

### 2. 配置版本控制

```bash
# .gitignore 中排除敏感配置
config/local.json
config/secrets.json
*.log
```

### 3. 配置备份策略

```python
import shutil
from datetime import datetime

def backup_config():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"config_backup_{timestamp}.json"
    shutil.copy("src/config/config.json", backup_path)
    print(f"配置已备份到: {backup_path}")
```

### 4. 环境隔离

```python
import os

# 根据环境加载不同配置
env = os.getenv('ENVIRONMENT', 'development')
config_file = f"configs/{env}.json"

if os.path.exists(config_file):
    config_manager.load_from_file(config_file)
```

通过合理的配置管理，可以大大提高系统的灵活性和可维护性。建议根据具体使用场景选择合适的配置方式。