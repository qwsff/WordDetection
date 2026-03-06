#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
违规内容检测API服务
基于FastAPI的RESTful API服务
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..core.detector import WordDetector
from ..config.manager import config_manager


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic模型定义
class DetectionRequest(BaseModel):
    text: str
    return_probability: bool = True
    return_embedding: bool = False

class BatchDetectionRequest(BaseModel):
    """批量检测请求模型"""
    texts: List[str] = Field(..., min_items=1, max_items=1000, 
                            description="待检测的文本列表")
    batch_size: int = Field(default=32, ge=1, le=128, 
                           description="批处理大小")

class DetectionResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: Optional[str] = None

# 初始化FastAPI应用
app = FastAPI(
    title="违规内容检测API",
    description="基于语义理解的中文违规内容检测服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局检测器实例
detector = None

def initialize_detector():
    """初始化检测器"""
    global detector
    try:
        # 获取配置
        model_config = config_manager.get_model_config()
        detection_config = config_manager.get_detection_config()
        
        config = {
            'model_path': model_config.get('embedding_model_path'),
            'classifier_path': model_config.get('classifier_save_path'),
            'threshold': detection_config.get('threshold', 0.5)
        }
        
        detector = WordDetector(config)
        if detector.load_models():
            logger.info("检测器初始化成功")
        else:
            logger.error("检测器初始化失败")
    except Exception as e:
        logger.error(f"初始化检测器时发生错误: {e}")

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("正在启动违规内容检测API服务...")
    initialize_detector()

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "违规内容检测API服务",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "detector_loaded": detector.is_loaded if detector else False,
        "model_info": detector.get_model_info() if detector else {}
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_forbidden_content(request: DetectionRequest):
    """
    检测单个文本是否违规
    
    Args:
        request: 检测请求
        
    Returns:
        检测结果
    """
    if not detector or not detector.is_loaded:
        raise HTTPException(status_code=503, detail="检测器未准备好")
    
    try:
        result = detector.predict(
            request.text,
            return_probability=request.return_probability,
            return_embedding=request.return_embedding
        )
        
        return DetectionResponse(
            success=True,
            data=result,
            message="检测完成"
        )
        
    except Exception as e:
        logger.error(f"检测过程中发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_detect")
async def batch_detect(request: BatchDetectionRequest):
    """批量文本检测"""
    try:
        if not detector.is_loaded:
            raise HTTPException(status_code=503, detail="检测器未初始化")
        
        # 使用批处理预测
        results = detector.batch_predict(
            request.texts, 
            batch_size=getattr(request, 'batch_size', 32)
        )
        
        return {
            "success": True,
            "data": results,
            "message": "批量检测完成"
        }
        
    except Exception as e:
        logger.error(f"批量检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_current_config():
    """获取当前配置"""
    return {
        "success": True,
        "data": config_manager.config.all_config
    }

def main():
    """主函数"""
    # 验证配置
    errors = config_manager.validate_config()
    if errors:
        logger.error("配置验证失败:")
        for error in errors:
            logger.error(f"  - {error}")
        return
    
    # 获取API配置
    api_config = config_manager.get_api_config()
    
    logger.info("启动违规内容检测API服务...")
    logger.info(f"监听地址: {api_config.get('host')}:{api_config.get('port')}")
    
    # 修复模块导入路径
    import sys
    from pathlib import Path
    
    # 添加项目根目录到Python路径
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    uvicorn.run(
        "src.api.detection_api:app",  # 修正模块路径
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('reload', True),
        workers=api_config.get('workers', 1)
    )

if __name__ == "__main__":
    main()