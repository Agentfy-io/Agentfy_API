import asyncio
import traceback
import aiofiles
import aiohttp
from typing import Union
from urllib.parse import urlparse

import os

from app.config import settings
from app.utils.logger import setup_logger
from dotenv import load_dotenv

# 加载 .env 文件 | Load .env file
load_dotenv()

# 设置日志记录器 | Set up logger
logger = setup_logger(__name__)


class WhisperLemonFox:
    def __init__(self, lemon_fox_api_key: str = None):
        self.logger = logger
        self.lemonfox_url = "https://api.lemonfox.ai"
        self.lemonfox_api_key = lemon_fox_api_key or settings.LEMON_FOX_API_KEY
        self.headers = {
            "User-Agent": "Agentfy.io/1.0.0",
            "Authorization": f"Bearer {self.lemonfox_api_key}",
        }
        self.__check_api_key()

    def __check_api_key(self):
        if not self.lemonfox_api_key:
            raise RuntimeError("Missing Lemon Fox API Key")

    @staticmethod
    def is_http_url(file_path: str) -> bool:
        try:
            parsed_result = urlparse(file_path)
            return parsed_result.scheme in ["http", "https"] and bool(parsed_result.netloc)
        except ValueError:
            return False

    async def transcriptions(
            self,
            file: str,
            response_format: str = "verbose_json",
            speaker_labels: bool = False,
            prompt: str = "",
            language: str = "",
            callback_url: str = "",
            translate: bool = False,
            timestamp_granularities: list = None,
            timeout: int = 60,
    ) -> Union[dict, str]:
        url = f"{self.lemonfox_url}/v1/audio/transcriptions"
        timeout_obj = aiohttp.ClientTimeout(total=timeout)

        # 构建基本数据
        data = {
            "response_format": response_format,
            "speaker_labels": speaker_labels,
            "prompt": prompt,
            "language": language,
            "callback_url": callback_url,
            "translate": translate,
            "timestamp_granularities": timestamp_granularities if timestamp_granularities else [],
        }

        try:
            async with aiohttp.ClientSession(headers=self.headers, timeout=timeout_obj) as session:
                if self.is_http_url(file):
                    # URL方式：直接在数据中传递文件URL
                    data["file"] = file
                    async with session.post(url, json=data) as response:
                        await self._check_response(response)
                        return await self._process_response(response, response_format)
                else:
                    # 本地文件方式：使用 FormData
                    form_data = aiohttp.FormData()
                    # 添加其他参数到 FormData
                    for key, value in data.items():
                        form_data.add_field(key, str(value))

                    # 添加文件
                    async with aiofiles.open(file, 'rb') as f:
                        file_content = await f.read()
                        form_data.add_field('file',
                                            file_content,
                                            filename=file.split('/')[-1],
                                            content_type='application/octet-stream')

                    async with session.post(url, data=form_data) as response:
                        await self._check_response(response)
                        return await self._process_response(response, response_format)

        except aiohttp.ClientError as e:
            self.logger.error(f"Network error occurred: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise

    async def _check_response(self, response: aiohttp.ClientResponse):
        """检查响应状态"""
        if response.status >= 400:
            error_text = await response.text()
            raise aiohttp.ClientResponseError(
                response.request_info,
                response.history,
                status=response.status,
                message=f"API request failed: {error_text}"
            )

    async def _process_response(self, response: aiohttp.ClientResponse, response_format: str):
        """处理响应内容"""
        self.logger.info(f"Transcription completed, status: {response.status}, format: {response_format}")
        if response_format in ["json", "verbose_json"]:
            return await response.json()
        return await response.text()


# 测试代码
async def run_test():
    wisper = WhisperLemonFox()
    file = "https://output.lemonfox.ai/wikipedia_ai.mp3"

    try:
        result = await wisper.transcriptions(
            file=file,
            response_format="verbose_json",
            speaker_labels=False,
            prompt="",
            language="",
            callback_url="",
            translate=False,
            timestamp_granularities=None,
            timeout=60
        )
        print("Transcription result:\n", result)
    except Exception as e:
        print("An error occurred:\n", e)
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_test())