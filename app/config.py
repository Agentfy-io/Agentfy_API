# -*- coding: utf-8 -*-
"""
@file: config.py
@desc: FastAPI应用配置
"""
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
    LEMONFOX_API_KEY: Optional[str] = Field(None, env="LEMONFOX_API_KEY")
    LOVO_API_KEY: Optional[str] = Field(None, env="LOVO_API_KEY")
    ELEVENLABS_API_KEY: Optional[str] = Field(None, env="ELEVENLABS_API_KEY")

    # TikHub API 配置
    TIKHUB_BASE_URL: str = Field("https://api.tikhub.io", env="TIKHUB_BASE_URL")

    #report 路径
    REPORT_PATH: str = Field("reports", env="REPORT_PATH")

    # 服务器设置
    HOST: str = Field("64.23.158.208", env="HOST")
    PORT: int = Field(80, env="PORT")
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

    # 静态文件配置参数
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    STATIC_DIR: str = "static"
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    SUPPORTED_VIDEO_FORMATS: list = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    CLEANUP_INTERVAL: int = 1800  # 一小时，单位秒


    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建全局设置对象
settings = Settings()