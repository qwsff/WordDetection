#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器主类
提供面向对象的配置管理接口
"""

import os
from typing import Any, Dict
from .config_manager import load_config, save_config, get_config_value

class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._config = None
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取配置字典"""
        if self._config is None:
            self._config = load_config(self.config_path)
        return self._config
    
    def reload(self):
        """重新加载配置"""
        self._config = load_config(self.config_path)
    
    def save(self):
        """保存配置"""
        save_config(self._config, self.config_path)
    
    def get(self, key_path: str, default=None) -> Any:
        """获取配置值"""
        return get_config_value(self.config, key_path, default)
    
    def set(self, key_path: str, value: Any):
        """设置配置值"""
        keys = key_path.split('.')
        current = self._config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value

# 创建默认实例
config_manager = ConfigManager()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器兼容模块
为保持向后兼容性而创建的包装模块
"""

# 尝试从本地导入配置管理器
try:
    from .config_manager import create_config, validate_config, update_config, print_config_summary, interactive_config, main
    
    # 为了向后兼容，导出相同的接口
    __all__ = [
        'create_config', 
        'validate_config', 
        'update_config', 
        'print_config_summary', 
        'interactive_config', 
        'main'
    ]
    
except ImportError as e:
    raise ImportError(f"无法加载配置管理模块: {e}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件管理工具
提供配置文件的创建、验证、更新等功能
"""

import argparse
import json
import os
from typing import Dict, Any
from config工具.config import DEFAULT_CONFIG, save_config, load_config, get_config_value

def create_config(config_path: str = "config.json"):
    """创建新的配置文件"""
    print("🔧 创建配置文件...")
    save_config(DEFAULT_CONFIG, config_path)
    print(f"✅ 配置文件已创建: {config_path}")
    print_config_summary()

def validate_config(config_path: str = "config.json"):
    """验证配置文件"""
    print("🔍 验证配置文件...")
    try:
        config = load_config(config_path)
        print("✅ 配置文件验证通过")
        
        # 检查必要配置项
        required_paths = [
            "model.path",
            "data.train_file", 
            "output.classifier_path"
        ]
        
        for path in required_paths:
            value = get_config_value(config, path)
            if value:
                print(f"   ✓ {path}: {value}")
            else:
                print(f"   ✗ {path}: 未配置")
                
    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")

def update_config(key_path: str, value: str, config_path: str = "config.json"):
    """更新配置项"""
    print(f"✏️  更新配置项: {key_path} = {value}")
    try:
        config = load_config(config_path)
        
        # 解析键路径
        keys = key_path.split('.')
        current = config
        
        # 导航到目标位置
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 设置值（尝试转换类型）
        try:
            # 尝试转换为数字
            if '.' in value:
                current[keys[-1]] = float(value)
            else:
                current[keys[-1]] = int(value)
        except ValueError:
            # 转换为布尔值
            if value.lower() in ['true', 'false']:
                current[keys[-1]] = value.lower() == 'true'
            else:
                current[keys[-1]] = value
        
        # 保存配置
        save_config(config, config_path)
        print("✅ 配置已更新")
        
    except Exception as e:
        print(f"❌ 更新配置失败: {e}")

def print_config_summary(config_path: str = "config.json"):
    """打印配置摘要"""
    try:
        config = load_config(config_path)
        print("\n📋 当前配置摘要:")
        print("=" * 50)
        
        # 模型配置
        print("🤖 模型配置:")
        print(f"   路径: {get_config_value(config, 'model.path')}")
        print(f"   设备: {get_config_value(config, 'model.device')}")
        print(f"   最大长度: {get_config_value(config, 'model.max_length')}")
        
        # 数据配置
        print("\n📊 数据配置:")
        print(f"   训练文件: {get_config_value(config, 'data.train_file')}")
        print(f"   编码: {get_config_value(config, 'data.encoding')}")
        
        # 训练配置
        print("\n🏋️ 训练配置:")
        print(f"   随机种子: {get_config_value(config, 'training.seed')}")
        print(f"   测试集比例: {get_config_value(config, 'training.test_size')}")
        
        # API配置
        print("\n🌐 API配置:")
        print(f"   主机: {get_config_value(config, 'api.host')}")
        print(f"   端口: {get_config_value(config, 'api.port')}")
        print(f"   热重载: {get_config_value(config, 'api.reload')}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 读取配置失败: {e}")

def interactive_config():
    """交互式配置向导"""
    print("🧙 交互式配置向导")
    print("=" * 30)
    
    config = DEFAULT_CONFIG.copy()
    
    # 模型配置
    print("\n🤖 模型配置")
    model_path = input(f"模型路径 [{config['model']['path']}]: ").strip()
    if model_path:
        config['model']['path'] = model_path
    
    device = input(f"运行设备 (cpu/cuda) [{config['model']['device']}]: ").strip()
    if device:
        config['model']['device'] = device
    
    # 数据配置
    print("\n📊 数据配置")
    train_file = input(f"训练数据文件 [{config['data']['train_file']}]: ").strip()
    if train_file:
        config['data']['train_file'] = train_file
    
    # API配置
    print("\n🌐 API配置")
    port = input(f"服务端口 [{config['api']['port']}]: ").strip()
    if port:
        try:
            config['api']['port'] = int(port)
        except ValueError:
            print("⚠️  端口必须是数字，使用默认值")
    
    # 保存配置
    save_config(config)
    print("\n✅ 配置已完成！")

def main():
    parser = argparse.ArgumentParser(description="配置文件管理工具")
    parser.add_argument("action", choices=['create', 'validate', 'update', 'show', 'interactive'], 
                       help="操作类型")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--key", help="配置键路径 (用于update)")
    parser.add_argument("--value", help="配置值 (用于update)")
    
    args = parser.parse_args()
    
    if args.action == 'create':
        create_config(args.config)
    elif args.action == 'validate':
        validate_config(args.config)
    elif args.action == 'update':
        if not args.key or not args.value:
            print("❌ update操作需要指定 --key 和 --value 参数")
            return
        update_config(args.key, args.value, args.config)
    elif args.action == 'show':
        print_config_summary(args.config)
    elif args.action == 'interactive':
        interactive_config()

if __name__ == "__main__":
    main()