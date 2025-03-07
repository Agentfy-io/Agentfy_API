import json
import traceback
from typing import Dict, Any, Optional

import pandas as pd
import requests
from app.config import settings
from app.utils.logger import setup_logger
from app.core.exceptions import ExternalAPIError

# 设置日志记录器
logger = setup_logger(__name__)


class Genny:
    """Genny API客户端封装类"""
    def __init__(self, lovo_api_key: Optional[str] = None):
        """
        初始化Genny客户端

        Args:
            lovo_api_key: Genny API密钥，如果未提供则从配置文件中读取
        """
        # 设置Genny API Key
        self.lovo_api_key = lovo_api_key or settings.LOVO_API_KEY

        if not self.lovo_api_key:
            logger.warning("未提供Genny API密钥，Genny功能将不可用")

        self.voice_query_url = "https://api.genny.lovo.ai/api/v1/speakers"
        self.gen_voice_url = "https://api.genny.lovo.ai/api/v1/tts/async"

    async def get_speakers(self, gender: str, age: str, language: str = "zh-CN") -> Dict[str, Any]:
        """
        获取Genny支持的发音人ID

        Returns:
            返回发音人列表（dict）
        """
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-API-KEY": self.lovo_api_key,
        }

        try:
            response = requests.get(self.voice_query_url, headers=headers)
            response.raise_for_status()

            response = response.json()
            speakers_list = response.get("data", [])
            speakers_df = pd.DataFrame(speakers_list)
            speakers_df = speakers_df[
                (speakers_df["gen