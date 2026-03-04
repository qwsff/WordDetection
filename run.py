#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目启动脚本
提供便捷的命令行接口来运行不同的功能
"""

import argparse
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    parser = argparse.ArgumentParser(description='WordDetection 项目管理工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data-path', type=str, help='训练数据路径')
    train_parser.add_argument('--model-path', type=str, help='嵌入模型路径')
    train_parser.add_argument('--save-path', type=str, help='模型保存路径')
    
    # API启动命令 - 添加模式选择
    api_parser = subparsers.add_parser('api', help='启动API服务')
    api_parser.add_argument('--mode', choices=['binary', 'multiclass'], default='binary', 
                           help='API模式: binary(二分类) 或 multiclass(多分类)')
    api_parser.add_argument('--host', type=str, default='0.0.0.0', help='主机地址')
    api_parser.add_argument('--port', type=int, default=8000, help='端口号')
    api_parser.add_argument('--local-model-only', action='store_true', 
                           help='仅使用本地模型，不从HF下载')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='运行测试')
    
    # 模型管理命令
    model_parser = subparsers.add_parser('model', help='模型管理')
    model_parser.add_argument('action', choices=['download', 'verify', 'info'], 
                             help='操作类型')
    model_parser.add_argument('--model-name', type=str, default='BAAI/bge-small-zh-v1.5',
                             help='模型名称')
    model_parser.add_argument('--local-path', type=str, 
                             help='本地模型路径')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from src.training.trainer import BinaryTrainer
        from src.config.manager import config_manager
        
        # 更新配置
        if args.data_path:
            config_manager.config.set('training.data_path', args.data_path)
        if args.model_path:
            config_manager.config.set('model.embedding_model_path', args.model_path)
        if args.save_path:
            config_manager.config.set('model.classifier_save_path', args.save_path)
        
        # 开始训练
        trainer = BinaryTrainer()
        result = trainer.train()
        
        if result['success']:
            print("✅ 训练成功!")
            print(f"📊 结果: {result['metrics']}")
        else:
            print("❌ 训练失败!")
            print(f"📝 错误: {result['message']}")
            
    elif args.command == 'api':
        import uvicorn
        from src.config.manager import config_manager
        
        # 设置本地模型优先
        if args.local_model_only:
            os.environ['SENTENCE_TRANSFORMERS_OFFLINE'] = '1'
            print("🔧 已设置为仅使用本地模型模式")
        
        # 更新API配置
        os.environ['API_HOST'] = args.host
        os.environ['API_PORT'] = str(args.port)
        os.environ['API_MODE'] = args.mode
        
        # 根据模式选择不同的API模块
        if args.mode == 'binary':
            app_module = "src.api.detection_api:app"
            print(f"🚀 启动二分类违禁词检测API服务...")
        else:
            app_module = "src.api.multiclass_api:app"
            print(f"🚀 启动多分类违禁词检测API服务...")
        
        print(f"📍 监听地址: {args.host}:{args.port}")
        print(f"🔄 开发模式: {'开启' if True else '关闭'}")
        
        # 直接使用uvicorn启动
        uvicorn.run(
            app_module,
            host=args.host,
            port=args.port,
            reload=True,
            workers=1
        )
        
    elif args.command == 'test':
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
    elif args.command == 'model':
        if args.action == 'download':
            download_model(args.model_name, args.local_path)
        elif args.action == 'verify':
            verify_model(args.local_path or 'models/bge-small-zh-v1.5')
        elif args.action == 'info':
            show_model_info(args.local_path or 'models/bge-small-zh-v1.5')
            
    else:
        parser.print_help()

def download_model(model_name: str, local_path: str = None):
    """下载模型到本地"""
    try:
        from sentence_transformers import SentenceTransformer
        import os
        
        local_path = local_path or f"models/{model_name.split('/')[-1]}"
        
        print(f"📥 正在下载模型: {model_name}")
        print(f"💾 保存位置: {local_path}")
        
        # 下载模型
        model = SentenceTransformer(model_name)
        
        # 保存到本地
        os.makedirs(local_path, exist_ok=True)
        model.save(local_path)
        
        print("✅ 模型下载完成!")
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")

def verify_model(local_path: str):
    """验证本地模型完整性"""
    try:
        from sentence_transformers import SentenceTransformer
        import os
        
        model_path = Path(local_path)
        if not model_path.exists():
            print(f"❌ 模型路径不存在: {local_path}")
            return
            
        required_files = [
            'config.json',
            'modules.json', 
            'pytorch_model.bin',
            'tokenizer.json',
            'tokenizer_config.json'
        ]
        
        missing_files = []
        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ 模型文件不完整，缺少: {missing_files}")
        else:
            print("✅ 模型文件完整")
            
            # 尝试加载模型
            print("🔍 正在验证模型加载...")
            model = SentenceTransformer(str(model_path))
            print("✅ 模型加载成功")
            
    except Exception as e:
        print(f"❌ 模型验证失败: {e}")

def show_model_info(local_path: str):
    """显示模型信息"""
    try:
        import json
        from pathlib import Path
        
        model_path = Path(local_path)
        config_file = model_path / 'config.json'
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"📄 模型配置信息 ({local_path}):")
            print(json.dumps(config, indent=2, ensure_ascii=False))
        else:
            print(f"❌ 未找到模型配置文件: {config_file}")
            
    except Exception as e:
        print(f"❌ 获取模型信息失败: {e}")

if __name__ == '__main__':
    main()