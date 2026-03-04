# WordDetection - 中文违规内容检测系统

基于语义理解的中文违规内容实时检测系统，支持二分类和多分类模式，能够识别谐音、隐晦表达等传统关键词过滤难以发现的违规内容。

## 🌟 系统特性

- **双模式支持**：同时支持二分类（正常/违规）和多分类（5种细化类别）检测
- **语义理解**：基于[bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)嵌入模型，理解深层语义
- **高性能**：轻量级设计，CPU友好，单次检测延迟 < 500ms
- **灵活部署**：支持本地部署、Docker容器化等多种部署方式
- **统一管理**：通过`run.py`统一脚本管理训练、服务、测试等操作

## 📁 项目结构

```
WordDetection/
├── src/                    # 源代码目录
│   ├── api/               # API服务
│   │   ├── detection_api.py      # 二分类检测API
│   │   └── multiclass_api.py     # 多分类API
│   ├── core/              # 核心逻辑
│   │   └── detector.py           # 检测器核心类
│   ├── training/          # 训练模块
│   │   ├── trainer.py            # 二分类训练器
│   │   └── multiclass_trainer.py # 多分类训练器
│   ├── config/            # 配置管理
│   │   ├── config.json           # 主配置文件
│   │   ├── config.py             # 配置基类
│   │   ├── manager.py            # 配置管理器
│   │   └── config_manager.py     # 配置管理兼容模块
│   └── utils/             # 工具函数
│       └── helpers.py            # 辅助工具函数
├── models/                # 模型文件
│   ├── bge-small-zh-v1.5/        # 预训练嵌入模型
│   └── classifiers/              # 训练好的分类器（运行时生成）
├── data/                  # 数据文件
│   └── labeled_data.csv          # 标注数据示例
├── tests/                 # 测试目录
│   ├── test_basic.py             # 基础功能测试
│   └── test_detector.py          # 核心检测功能测试
├── examples/              # 示例代码
│   ├── complete_demo.py          # 完整功能演示
│   └── README.md                 # 示例说明
├── run.py                 # 统一启动脚本
├── requirements.txt       # 依赖包列表
├── setup.py               # 安装配置
├── pytest.ini            # 测试配置文件
├── .gitignore            # Git忽略文件
└── README.md              # 项目说明（本文档）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备训练数据

#### 二分类数据格式
项目已包含示例数据 `data/labeled_data.csv`，格式如下：

```csv
text,label
这是一条正常的评论内容,0
你真是个垃圾,1
这种说法太不合适了,1
我觉得这个观点很有道理,0
```

其中：
- `text`: 待检测的文本内容
- `label`: 标签（0=正常，1=违规）

#### 多分类数据格式
对于多分类模式，需要提供细化的类别标签：

```csv
text,category
这是一条正常的评论内容,normal
我要杀了你,threat
色情相关内容,porn
刷单返利信息,fraud
你这个蠢货,insult
```

支持的多分类类别：
- **normal**: 正常内容
- **threat**: 威胁内容（死亡威胁、人身威胁等）
- **porn**: 色情内容（色情、成人内容等）
- **fraud**: 欺诈内容（诈骗、虚假信息等）
- **insult**: 辱骂内容（侮辱、谩骂等）

### 3. 训练模型

```bash
# 使用统一启动脚本训练（推荐）
python run.py train

# 或者指定自定义路径
python run.py train --data-path /path/to/data.csv --model-path /path/to/model

# 多分类训练（如果数据包含category列）
python run.py train --data-path data/multiclass_data.csv
```

### 4. 启动API服务

推荐使用统一启动脚本：

```bash
# 启动二分类API（默认）
python run.py api

# 启动多分类API
python run.py api --mode multiclass

# 自定义配置
python run.py api --host 0.0.0.0 --port 8000 --mode binary

# 离线模式（仅使用本地模型）
python run.py api --local-model-only
```

### 5. 测试API

#### 二分类检测
```bash
# 单个文本检测
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: application/json" \
     -d '{"text": "测试文本内容", "return_probability": true}'

# 批量检测
curl -X POST "http://localhost:8000/batch_detect" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["文本1", "文本2", "文本3"], "batch_size": 32}'

# 健康检查
curl "http://localhost:8000/health"
```

#### 多分类检测
```bash
# 单个文本检测
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: application/json" \
     -d '{"text": "我要杀了你", "return_probabilities": true}'

# 获取支持的分类列表
curl "http://localhost:8000/categories"

# 健康检查（包含分类信息）
curl "http://localhost:8000/health"
```

## ⚙️ 配置管理

系统使用统一的配置管理模块，配置文件位于 `src/config/config.json`。

### 主要配置项

```json
{
  "model": {
    "embedding_model_path": "models/bge-small-zh-v1.5",
    "classifier_save_path": "models/classifiers/forbidden_classifier.pkl",
    "multiclass_classifier_path": "models/classifiers/multiclass_classifier.pkl"
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
  }
}
```

### 环境变量覆盖

可以通过环境变量临时覆盖配置：

```bash
export API_PORT=8080
export API_HOST=127.0.0.1
export MODEL_EMBEDDING_MODEL_PATH=/custom/path/to/model
export TRAINING_DATA_PATH=/custom/data/path
```

## 🔧 开发指南

### 项目安装（开发模式）

```bash
pip install -e .
```

### 运行测试

```bash
# 运行所有测试
python run.py test

# 或直接使用pytest
python -m pytest tests/

# 运行特定测试文件
python -m pytest tests/test_detector.py -v

# 生成覆盖率报告
python -m pytest tests/ --cov=src --cov-report=html

# 运行带标记的测试
python -m pytest tests/ -m "not performance"  # 跳过性能测试
```

### 代码格式化

```bash
black src/
flake8 src/
```

### 模型管理工具

```bash
# 查看模型信息
python run.py model info

# 验证模型完整性
python run.py model verify

# 下载模型（如果需要）
python run.py model download
```

## 📊 API接口文档

启动服务后访问 `http://localhost:8000/docs` 查看完整的API文档。

### 二分类API接口

- `GET /` - 根路径，服务状态检查
- `GET /health` - 健康检查
- `POST /detect` - 单文本检测
- `POST /batch_detect` - 批量文本检测
- `GET /config` - 获取当前配置

### 多分类API接口

- `GET /` - 根路径，服务状态检查
- `GET /health` - 健康检查（包含分类信息）
- `POST /detect` - 单文本多分类检测
- `POST /batch_detect` - 批量多分类检测
- `GET /categories` - 获取支持的分类列表
- `GET /config` - 获取当前配置

## 🛠️ 技术架构

### 核心组件

1. **嵌入模型**: 使用 `bge-small-zh-v1.5` 进行中文文本语义编码
2. **分类器**: 支持二分类和多分类模型
3. **API服务**: 基于FastAPI的高性能RESTful API
4. **配置管理**: 统一的配置管理系统

### 系统架构图

```
[客户端] → HTTP请求 → [FastAPI服务] → 加载模型 → [bge-small-zh-v1.5嵌入 + 分类器] → 返回JSON结果
```

## 🔒 安全说明

⚠️ **重要提醒**：
- 生产环境中建议添加身份验证机制
- 敏感配置不应提交到版本控制系统
- 定期更新训练数据以适应新的违规模式
- 建议在HTTPS环境下部署API服务

### 开发环境设置

```bash
git clone https://gitee.com/fsjhdf/WordDetection.git
cd WordDetection
pip install -e ".[dev]"
```

## 📈 性能优化建议

1. **硬件配置**：推荐4核CPU以上，8GB内存
2. **批处理**：大量文本检测时使用批量接口
3. **缓存策略**：对高频检测内容实施缓存
4. **负载均衡**：生产环境建议使用负载均衡

## 🆘 常见问题解答

### Q: 模型加载失败怎么办？
A: 检查 `bge-small-zh-v1.5` 文件夹是否存在且完整，或使用 `--local-model-only` 参数

### Q: 训练数据格式有什么要求？
A: CSV格式，包含 `text` 列和 `label` 或 `category` 列，编码为UTF-8

### Q: 如何提高检测准确率？
A: 增加训练数据量，特别是边界案例和误判案例，建议每个类别至少50条样本

### Q: 支持哪些部署方式？
A: 支持本地部署、Docker容器化部署、云服务器部署等多种方式

### Q: 多分类和二分类有什么区别？
A: 多分类提供更细粒度的违规内容分类，而二分类只区分正常和违规

## 📝 更新日志

### v2.0.0 (最新版本)
- ✅ 新增多分类检测功能
- ✅ 支持5种细化违规内容分类
- ✅ 统一的命令行管理工具
- ✅ 智能模型加载策略
- ✅ 完善的配置管理系统
- ✅ 详细的测试套件

### v1.0.0 (初始版本)
- ✅ 基础二分类违禁内容检测
- ✅ 基于语义理解的内容识别
- ✅ RESTful API接口服务

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

MIT License

## 🙏 致谢

- [sentence-transformers](https://www.sbert.net/) - 嵌入模型框架
- [FastAPI](https://fastapi.tiangolo.com/) - API框架
- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [BGE](https://huggingface.co/BAAI/bge-small-zh-v1.5) - 中文语义嵌入模型