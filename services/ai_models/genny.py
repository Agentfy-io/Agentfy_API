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
        self.gen_voice_url = "https://api.genny.lovo.ai/api/v1/tts/sync"

    async def get_speakers(self, gender: str, age: str, language: str = "zh-CN") -> Dict[str, Any]:
        """
        获取Genny支持的发音人ID

        Returns:
            返回发音人列表（dict）
        """
        if gender not in ["male", "female"]:
            raise ValueError("Invalid gender value, either male or female")

        if age not in ["child", "teen", "adult", "senior"]:
            raise ValueError("Invalid age value, either child, young_adult, mature_adult, teen, or old")

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-API-KEY": self.lovo_api_key
        }

        response = requests.get(self.voice_query_url, headers=headers)
        response.raise_for_status()

        speakers = response.json()

        # use panda to filter the speakers
        speakers_df = pd.DataFrame(speakers['data'])
        speakers_df = speakers_df[(speakers_df['gender'] == gender) & (speakers_df['ageRange'] == age) & (speakers_df['locale'] == language) & (speakers_df['speakerType'] == 'global')]

        return speakers_df.to_dict(orient='records')

    async def generate_voice(self, text: str, speaker_id: str, speed: int = 1) -> Dict[str, Any]:
        """
        生成Genny音频文件

        Args:
            text: 待转换的文本
            speaker_id: 说话者ID
            speed: 语速

        Returns:
            生成的音频文件信息
        """
        # 构建请求参数
        payload = {
            "speed": speed,
            "text": text,
            "speaker": speaker_id
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-API-KEY": self.lovo_api_key
        }

        # 发送请求
        try:
            response = requests.post(self.gen_voice_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to generate Genny voice: {str(e)}")
            raise ExternalAPIError("Failed to generate Genny voice")

