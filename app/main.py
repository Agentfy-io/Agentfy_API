# -*- coding: utf-8 -*-
"""
@file: main.py
@desc: FastAPI应用入口
@auth: Callmeiks
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import uvicorn
from dotenv import load_dotenv

from .api.routes import customer, auth, sentiment, video, audio, user
from .core.exceptions import CommentAPIException
from .utils.logger import setup_logger
from .dependencies import log_request_middleware

# 加载环境变量
load_dotenv()

# 设置日志
logger = setup_logger(__name__)

title = "Agentfy API - Any Data, Any Analysis, Any Generators"
description = f"""
--------------------------
## 评论分析
- **购买意向分析**：识别有购买意向的用户，并生成强烈意向购买客户列表。
- **评论舆情分析**：分析评论内容，了解观众的整体情绪。
- **与视频创作者的关系分析**：识别忠诚粉丝、普通观众、黑粉等用户类别。
- **Toxicity 分析（恶评/差评分析）**：
  - 识别恶意评论，包括辱骂、骚扰、自我推广等无用信息。
  - 识别产品相关的负面评论，如“商品不好用”或“商品未收到”等售后问题。
--------------------------
## 达人分析
- **用户基本信息分析**（如商家账户，提取其公司信息）。
- **发帖数据分析**：
    - 用户发帖趋势（默认 30 天）。
    - 用户发帖的视频时长分布。
    - 用户发帖时间分布。
    - 热门视频 Top 5。
    - 用户的广告/商业视频分析。
    - AI/VR 视频使用情况分析。
    - 最常用标签/话题 Top 20。
- **粉丝信息分析**
- **风险视频分析**：基于 TikTok 的字段判断（而非内容+标题）。
- **指定用户发帖的评论分析**。
--------------------------
## 视频分析
- **基础信息获取**：
  - 观看量、点赞数、评论数、转发数。
  - 视频长度、分辨率、上传日期等。
- **下载视频**。
- **转录内容分析（Whisper）**。
- **带货视频分析**：
  - 识别产品特性、价格信息、促销活动。
  - 竞争对手营销话术和价值主张分析。
- **内容创作分析**：
  - 识别创作者常用的语言模式和表达方式。
  - 自动生成视频摘要或提取重点内容。
- **帧内容分析**（用户可自定义关键帧间隔，如每 2 秒获取一帧）：
  - **使用 OpenCV 进行视频帧分析**。
  - **使用单独的图片模型进行二次分析（时间较长）**。
  - ChatGPT 生成视频脚本。
  - 追踪品牌曝光频率和方式。
  - 识别视频场景变化和叙事结构。
- **OCR 识别视频文本内容**：
  - 识别产品名称、标签、价格、促销信息。
  - 识别品牌名称、产品型号。
  - 识别视频中的网址、社交媒体账号、联系方式。
--------------------------
#### 赞助商家/合作品牌相关链接
- **🏠 Home**: [https://www.tikhub.io](https://www.tikhub.io)
- **👨‍💻 Github**: [https://github.com/TikHub](https://github.com/TikHub)
- **⚡ Documents (Swagger UI)**: [https://api.tikhub.io](https://api.tikhub.io)
- **🦊 Documents (Apifox UI)**: [https://docs.tikhub.io](https://docs.tikhub.io)
- **🍱 SDK**: [https://github.com/TikHub/TikHub-API-Python-SDK](https://github.com/TikHub/TikHub-API-Python-SDK)
- **📧 Support**: [Discord Server](https://discord.gg/aMEAS8Xsvz)

"""

# 创建 FastAPI 应用
app = FastAPI(
    title=title,
    description=description,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    # 配置Swagger UI显示
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,  # 隐藏模型
        "operationsSorter": "alpha",  # 按字母排序操作
        "tryItOutEnabled": True,  # 默认启用"Try it out"
        "displayRequestDuration": True,  # 显示请求持续时间
        "filter": True  # 启用过滤功能
    }
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境下应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 添加日志中间件
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    return await log_request_middleware(request, call_next)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # 添加安全定义
    openapi_schema["components"] = openapi_schema.get("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "TikHubBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API Key",
            "description": "输入您的TikHub API密钥"
        }
    }

    # 全局安全要求
    openapi_schema["security"] = [{"TikHubBearer": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# 覆盖默认的OpenAPI架构
app.openapi = custom_openapi

# 注册路由
# app.include_router(auth.router, prefix="/api/v1", tags=["认证"])
app.include_router(customer.router, prefix="/api/v1", tags=["购买客户分析"])
app.include_router(user.router, prefix="/api/v1", tags=["用户/达人分析"])
app.include_router(sentiment.router, prefix="/api/v1", tags=["评论舆情分析"])
app.include_router(video.router, prefix="/api/v1", tags=["视频全方位分析"])
app.include_router(audio.router, prefix="/api/v1", tags=["短视频脚本/音频生成"])
app.include_router(xhs.router, prefix="/api/v1", tags=["小红书生成"])

# 全局异常处理
@app.exception_handler(CommentAPIException)
async def comment_api_exception_handler(request: Request, exc: CommentAPIException):
    logger.error(f"API错误: {exc.detail}, 状态码: {exc.status_code}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": exc.error_type
            },
            "data": None,
            "meta": {
                "timestamp": exc.timestamp.isoformat(),
                "path": request.url.path
            }
        }
    )


@app.get("/", tags=["root"])
async def root():
    """将根路径重定向到API文档"""
    return RedirectResponse(url="/docs")


# 主函数
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "False").lower() == "true"

    print(f"Starting server at {host}:{port} with debug={debug}")

    # 启动服务器
    uvicorn.run(app, host=host, port=port, reload=debug)