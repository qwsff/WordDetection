#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WordDetection 完整功能演示脚本
展示项目的各项核心功能和使用方法
"""

import os
import sys
import json
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

def demo_configuration():
    """演示配置管理功能"""
    print("🔧 配置管理演示")
    print("=" * 50)
    
    try:
        from config.manager import config_manager
        
        # 显示当前配置
        print("当前配置:")
        config_data = config_manager.config.all_config
        print(json.dumps(config_data, indent=2, ensure_ascii=False))
        
        # 验证配置
        errors = config_manager.validate_config()
        if errors:
            print("配置验证发现问题:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("✅ 配置验证通过")
            
    except Exception as e:
        print(f"❌ 配置管理演示失败: {e}")
    
    print()

def demo_model_management():
    """演示模型管理功能"""
    print("📦 模型管理演示")
    print("=" * 50)
    
    # 验证模型完整性
    model_path = "models/bge-small-zh-v1.5"
    if os.path.exists(model_path):
        print(f"🔍 验证模型: {model_path}")
        
        required_files = [
            'config.json',
            'modules.json', 
            'pytorch_model.bin',
            'tokenizer.json'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ 模型文件不完整，缺少: {missing_files}")
        else:
            print("✅ 模型文件完整")
            
            # 尝试加载模型
            try:
                from sentence_transformers import SentenceTransformer
                print("🔍 正在验证模型加载...")
                model = SentenceTransformer(model_path)
                print("✅ 模型加载成功")
                print(f"   嵌入维度: {model.get_sentence_embedding_dimension()}")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
    else:
        print(f"❌ 模型路径不存在: {model_path}")
    
    print()

def demo_binary_detection():
    """演示二分类检测功能"""
    print("🎯 二分类检测演示")
    print("=" * 50)
    
    try:
        from core.detector import WordDetector
        from config.manager import config_manager
        
        # 初始化检测器
        model_config = config_manager.get_model_config()
        detection_config = config_manager.get_detection_config()
        
        config = {
            'model_path': model_config.get('embedding_model_path'),
            'classifier_path': model_config.get('classifier_save_path'),
            'threshold': detection_config.get('threshold', 0.5)
        }
        
        detector = WordDetector(config)
        
        if detector.load_models():
            print("✅ 检测器初始化成功")
            
            # 测试文本
            test_texts = [
                "这是一篇很好的文章，内容很有价值",
                "你真是个垃圾，说话太难听了",
                "我觉得这个观点不太合适",
                "谢谢你的分享，学到了很多"
            ]
            
            print("\n检测结果:")
            for text in test_texts:
                result = detector.predict(text, return_probability=True)
                status = "🔴 违规" if result["is_forbidden"] else "🟢 正常"
                probability = result.get("probability", 0)
                print(f"  {status} | '{text}' | 概率: {probability:.3f}")
        else:
            print("❌ 检测器初始化失败")
            
    except Exception as e:
        print(f"❌ 二分类检测演示失败: {e}")
    
    print()

def demo_multiclass_detection():
    """演示多分类检测功能"""
    print("🌈 多分类检测演示")
    print("=" * 50)
    
    try:
        from core.detector import MulticlassWordDetector
        from config.manager import config_manager
        
        # 初始化多分类检测器
        model_config = config_manager.get_model_config()
        detection_config = config_manager.get_detection_config()
        
        config = {
            'model_path': model_config.get('embedding_model_path'),
            'classifier_path': model_config.get('multiclass_classifier_path'),
            'threshold': detection_config.get('threshold', 0.5)
        }
        
        detector = MulticlassWordDetector(config)
        
        if detector.load_models():
            print("✅ 多分类检测器初始化成功")
            print(f"支持的类别: {getattr(detector, 'class_labels', [])}")
            
            # 测试文本
            test_texts = [
                "这是一篇很好的文章，内容很有价值",
                "你真是个垃圾，说话太难听了", 
                "我觉得这个观点不太合适",
                "这种表达方式我很反感"
            ]
            
            print("\n检测结果:")
            for text in test_texts:
                result = detector.predict(text, return_probability=True)
                predicted_class = result.get("predicted_class", "unknown")
                confidence = result.get("confidence", 0)
                print(f"  🎯 {predicted_class} | '{text}' | 置信度: {confidence:.3f}")
                
                # 显示概率分布
                if "probabilities" in result:
                    probs = result["probabilities"]
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"    Top3概率: {sorted_probs}")
        else:
            print("❌ 多分类检测器初始化失败")
            
    except Exception as e:
        print(f"❌ 多分类检测演示失败: {e}")
        print("💡 提示: 可能需要先训练多分类模型")
    
    print()

def demo_batch_processing():
    """演示批量处理功能"""
    print("⚡ 批量处理演示")
    print("=" * 50)
    
    try:
        from core.detector import WordDetector
        from config.manager import config_manager
        
        # 初始化检测器
        model_config = config_manager.get_model_config()
        config = {
            'model_path': model_config.get('embedding_model_path'),
            'classifier_path': model_config.get('classifier_save_path')
        }
        
        detector = WordDetector(config)
        
        if detector.load_models():
            # 大量测试文本
            test_texts = [
                "正常文本内容示例",
                "这是违规的表达方式",
                "中性的讨论内容",
                "不当的言辞表述",
                "积极正面的评价",
                "消极负面的言论"
            ] * 10  # 60条文本
            
            print(f"处理 {len(test_texts)} 条文本...")
            
            start_time = time.time()
            results = detector.batch_predict(test_texts, batch_size=32)
            end_time = time.time()
            
            # 统计结果
            forbidden_count = sum(1 for r in results if r["is_forbidden"])
            normal_count = len(results) - forbidden_count
            
            print(f"✅ 批量处理完成!")
            print(f"   处理时间: {end_time - start_time:.2f} 秒")
            print(f"   处理速度: {len(test_texts)/(end_time - start_time):.1f} 条/秒")
            print(f"   违规文本: {forbidden_count} 条")
            print(f"   正常文本: {normal_count} 条")
        else:
            print("❌ 检测器初始化失败")
            
    except Exception as e:
        print(f"❌ 批量处理演示失败: {e}")
    
    print()

def demo_performance_monitoring():
    """演示性能监控功能"""
    print("📈 性能监控演示")
    print("=" * 50)
    
    try:
        from core.detector import WordDetector
        from config.manager import config_manager
        
        # 初始化检测器
        model_config = config_manager.get_model_config()
        config = {
            'model_path': model_config.get('embedding_model_path'),
            'classifier_path': model_config.get('classifier_save_path')
        }
        
        detector = WordDetector(config)
        
        if detector.load_models():
            # 获取模型信息
            model_info = detector.get_model_info()
            print("模型信息:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
            
            # 性能测试
            test_text = "这是一个性能测试文本"
            
            # 单次预测时间测试
            times = []
            for i in range(10):
                start_time = time.time()
                detector.predict(test_text)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n性能测试结果 (10次平均):")
            print(f"  平均响应时间: {avg_time*1000:.2f} ms")
            print(f"  最快响应时间: {min_time*1000:.2f} ms") 
            print(f"  最慢响应时间: {max_time*1000:.2f} ms")
            
        else:
            print("❌ 检测器初始化失败")
            
    except Exception as e:
        print(f"❌ 性能监控演示失败: {e}")
    
    print()

def main():
    """主函数"""
    print("🚀 WordDetection 完整功能演示")
    print("=" * 60)
    print("本脚本将演示项目的核心功能和使用方法\n")
    
    # 按顺序执行各个演示
    demo_configuration()
    demo_model_management() 
    demo_binary_detection()
    demo_multiclass_detection()
    demo_batch_processing()
    demo_performance_monitoring()
    
    print("🎉 演示完成!")
    print("\n💡 提示:")
    print("  - 使用 'python run.py --help' 查看所有可用命令")
    print("  - 使用 'python run.py api' 启动API服务")
    print("  - 使用 'python run.py train' 训练模型")
    print("  - 查看 docs/ 目录了解更多使用信息")

if __name__ == "__main__":
    main()