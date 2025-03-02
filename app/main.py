import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import uvicorn
from dotenv import load_dotenv

from app.api.routes import comments, auth
from app.core.exceptions import CommentAPIException
from app.utils.logger import setup_logger
from app.dependencies import log_request_middleware

# 加载环境变量
load_dotenv()

# 设置日志
logger = setup_logger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="TikTok评论分析API",
    description="获取和分析TikTok视频评论，识别潜在客户",
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


# 自定义OpenAPI架构，添加安全定义
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
        "SessionAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-Session-ID",
            "description": "会话ID认证，在登录后获取"
        }
    }

    # 全局安全要求
    openapi_schema["security"] = [{"SessionAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# 覆盖默认的OpenAPI架构
app.openapi = custom_openapi

# 注册路由
app.include_router(auth.router, prefix="/api/v1", tags=["认证"])
app.include_router(comments.router, prefix="/api/v1", tags=["评论"])


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

    # 启动服务器
    uvicorn.run("app.main:app", host=host, port=port, reload=debug)