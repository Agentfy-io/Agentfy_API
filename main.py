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
from app.api.routes import customer, auth, sentiment, video, audio, user, xhs
from app.core.exceptions import CommentAPIException
from app.utils.logger import setup_logger
from app.dependencies import log_request_middleware

# 加载环境变量
load_dotenv()

# 设置日志
logger = setup_logger(__name__)

title = "Agentfy API"
description = f"""
### TikTok Features

#### 📝 Comment Analysis:
- **Purchase Intent**: Identify potential buyers.
- **Sentiment**: Analyze audience sentiment.
- **Creator-Follower Relationship**: Classify users (loyal fans, trolls, etc.).
- **Toxicity**: Detect malicious and negative comments.

#### 🕵️‍♂️ Influencer/User Analysis:
- **User Info**: Extract influencer/business account data.
- **Post Data**: Analyze post trends, video length, top videos, hashtags.
- **Risk Video**: Analyze content risk based on TikTok fields.

#### 📹 Video Analysis:
- **Metrics**: Views, likes, shares, video length.
- **Download & Transcription**: Download videos and transcribe content.
- **Frame Analysis**: Extract frames and analyze using OpenCV.
- **OCR**: Extract text in videos (product names, prices, etc.).

### Generators Features

#### 🎥 Short Video Script Generation:
- **Script Generation**: Generate scripts for short videos.
- **Audio Generation**: Generate audio for short videos using your own voice or any voice.
- **Xiaohongshu Post Generation**: Generate Xiaohongshu posts based on Douyin videos. （Claude API key required）

------
### TikTok agent功能

#### 📝 评论分析：
- **购买意图**：识别潜在购买用户。
- **舆情分析**：分析观众情绪。
- **与创作者关系**：分类用户（忠实粉丝、普通观众黑粉等）。
- **恶意评论**：识别负面与恶意评论。

#### 🕵️‍♂️ 达人分析：
- **用户信息**：提取商家账户数据。
- **发帖数据**：分析发帖趋势、视频时长、热门视频、标签。
- **风险视频**：根据 TikTok 字段分析内容风险。
- **评论分析**：分析特定用户的评论。

#### 📹 视频分析：
- **基础数据**：观看量、点赞数、分享数、视频时长。
- **下载与转录**：下载视频并转录内容。
- **带货分析**：识别产品特点、价格和促销。
- **创作分析**：分析语言模式，自动生成摘要。
- **帧分析**：提取帧并使用 OpenCV 分析。
- **OCR**：提取视频中的文字（产品名称、价格等）。

#### 生成器功能
- **短视频脚本生成**：关键词生成短视频脚本。
- **音频生成**：使用您自己的声音或任何声音生成短视频的音频。
- **小红书笔记生成**：根据抖音视频生成小红书笔记/根据赛道关键词生成小红书笔记。（claude API key必填）

--------------------------

#### 🌐 Sponsor Links/相关链接

- **👨‍💻 Agentfy Github**: [https://github.com/Agentfy-io/Agentfy](https://github.com/Agentfy-io/Agentfy)
- **🏠 TikHub Home**: [https://www.tikhub.io](https://www.tikhub.io)
- **👨‍💻 TikHub Github**: [https://github.com/TikHub](https://github.com/TikHub)
- **⚡ TikHub Documents (Swagger UI)**: [https://api.tikhub.io](https://api.tikhub.io)
- **🦊 TikHub Documents (Apifox UI)**: [https://docs.tikhub.io](https://docs.tikhub.io)
- **📧 TikHub Support**: [Discord Server](https://discord.gg/aMEAS8Xsvz)

-------------------------------

### 🔐 Authorization/鉴权

请根据需要在Authorization中添加API密钥。

- **TikHub API Key**: [https://www.user.tikhub.io](https://www.user.tikhub.io) (Required/必填)
- **OpenAI API Key**: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)(Required/必填)
- **LemonFox API Key**: [https://lemonfox.ai](https://lemonfox.ai) (Optional/选填)
- **ElevenLabs API Key**: [https://beta.elevenlabs.io](https://beta.elevenlabs.io) (Optional/选填)

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

    openapi_schema["components"] = openapi_schema.get("components", {})

    openapi_schema["components"]["securitySchemes"] = {
        "TikHub": {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization",
            "description": "请输入 TikHub API Key， https://www.user.tikhub.io"
        },
        "OpenAI": {
            "type": "apiKey",
            "in": "header",
            "name": "OpenAI-Authorization",  # 👈Header名字（OpenAI-Authorization）
            "description": "请输入 OpenAI API Key, https://platform.openai.com/account/api-keys"
        },
        "Claude": {
            "type": "apiKey",
            "in": "header",
            "name": "Claude-Authorization",
            "description": "请输入Claude API Key (如果有小红书生成需求)"
        },
        "LemonFox": {
            "type": "apiKey",
            "in": "header",
            "name": "LemonFox-Authorization",
            "description": "请输入 LemonFox API Key (如果有视频音频转文字需求)"
        },
        "ElevenLabs":{
            "type": "apiKey",
            "in": "header",
            "name": "ElevenLabs-Authorization",
            "description": "请输入 ElevenLabs API Key (如果有生成音频需求)"
        }

    }

    openapi_schema["security"] = [
        {"TikHub": []},
        {"OpenAI": []},
        {"Claude": []},
        {"LemonFox": []},
        {"ElevenLabs": []}
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema



# 覆盖默认的OpenAPI架构
app.openapi = custom_openapi

# 注册路由
# app.include_router(auth.router, prefix="/api/v1", tags=["认证"])
app.include_router(customer.router, prefix="/api/v1", tags=["Customers"])
app.include_router(user.router, prefix="/api/v1", tags=["Influencers"])
app.include_router(sentiment.router, prefix="/api/v1", tags=["Comments"])
app.include_router(video.router, prefix="/api/v1", tags=["Videos"])
app.include_router(audio.router, prefix="/api/v1", tags=["Generators"])
app.include_router(xhs.router, prefix="/api/v1", tags=["Xiaohongshu"])

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