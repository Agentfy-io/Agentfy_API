import os
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

# 确保环境变量已加载
load_dotenv()


class Settings(BaseSettings):
    """应用程序配置设置"""

    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    TIKHUB_API_KEY: Optional[str] = Field(None, env="TIKHUB_API_KEY")

    # TikHub API 配置
    TIKHUB_BASE_URL: str = Field("https://api.tikhub.io", env="TIKHUB_BASE_URL")

    # 服务器设置
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")
    DEBUG: bool = Field(False, env="DEBUG")

    # 日志设置
    LOG_LEVEL: str = Field("info", env="LOG_LEVEL")

    # API 行为设置
    DEFAULT_BATCH_SIZE: int = Field(30, env="DEFAULT_BATCH_SIZE")
    DEFAULT_CONCURRENCY: int = Field(5, env="DEFAULT_CONCURRENCY")
    MAX_BATCH_SIZE: int = Field(100, env="MAX_BATCH_SIZE")

    # AI 模型默认设置
    DEFAULT_AI_MODEL: str = Field("gpt-4o-mini", env="DEFAULT_AI_MODEL")
    DEFAULT_TEMPERATURE: float = Field(0.7, env="DEFAULT_TEMPERATURE")
    DEFAULT_MAX_TOKENS: int = Field(15000, env="DEFAULT_MAX_TOKENS")

    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建全局设置对象
settings = Settings()