import asyncio
import os
import uuid
from typing import Dict, Any, Optional, List

from elevenlabs import ElevenLabs
from app.config import settings
from app.utils.logger import setup_logger
from app.core.exceptions import ExternalAPIError

# 设置日志记录器
logger = setup_logger(__name__)


class ElevenLabsClient:
    """ElevenLabs API客户端封装类"""

    def __init__(self, api_key: Optional[str] = None, output_dir: str = "audio_files"):
        """
        初始化ElevenLabs客户端

        Args:
            api_key: ElevenLabs API密钥，如果未提供则从配置文件中读取
            output_dir: 保存音频文件的目录
        """
        # 设置ElevenLabs API Key
        self.api_key = api_key or settings.ELEVENLABS_API_KEY

        if not self.api_key:
            logger.warning("未提供ElevenLabs API密钥，ElevenLabs功能将不可用")

        # 初始化SDK
        self.client = ElevenLabs(api_key=self.api_key)

        # 设置输出目录
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    async def get_voices(
            self,
            language: str = "en",
            gender: str = "male",
            age: str = "middle-aged",
            voice_id: Optional[str] = None
    ) -> List[str]:
        """
        获取所有可用声音

        Args:
            language: 语音语言
            gender: 性别 (male/female)
            age: 年龄段 (young/middle-aged/old)
            voice_id: 特定声音ID（如果提供则直接返回）

        Returns:
            匹配条件的声音ID列表
        """
        # 如果提供了特定voice_id，直接返回
        if voice_id:
            return [voice_id]

        if gender not in ["male", "female"]:
            raise ValueError("Invalid gender value, either male or female")

        if age not in ["young", "middle-aged", "old"]:
            raise ValueError("Invalid age value, either young, middle-aged, or old")

        try:
            # 获取所有声音
            voices = self.client.voices.get_all()
            matched_voice_ids = []

            # 筛选匹配条件的声音
            for voice in voices.voices:
                labels = voice.labels
                if not labels:
                    continue

                if (labels['gender'] == gender and
                        labels['age'] == age and
                        voice.fine_tuning.language == language):
                    matched_voice_ids.append(voice.voice_id)

            return matched_voice_ids

        except Exception as e:
            logger.error(f"获取ElevenLabs声音列表失败: {str(e)}")
            raise ExternalAPIError("获取ElevenLabs声音列表失败")

    async def add_voice(
            self,
            name: str,
            files: Optional[List[str]] = None,
            description: Optional[str] = None,
            labels: Optional[Dict[str, str]] = None
    ) -> str:
        """
        添加新声音（声音克隆）

        Args:
            name: 声音名称
            files: 音频样本文件路径列表
            description: 声音描述（可选）
            labels: 标签字典（可选）

        Returns:
            创建的声音ID
        """
        try:
            # 检查文件路径
            audio_files = []

            if files:
                valid_extensions = (".mp3", ".mp4", ".webm", ".amr")

                for file_path in files:
                    # 检查文件格式
                    if not str(file_path).lower().endswith(valid_extensions):
                        raise ValueError(
                            f"Invalid audio file format for {file_path}, "
                            f"only mp3, MP4, webm, amr are supported"
                        )

                    # 检查文件是否存在
                    if not os.path.exists(file_path):
                        raise ValueError(f"File does not exist: {file_path}")

                    # 读取文件内容
                    with open(file_path, "rb") as f:
                        audio_files.append(f.read())

            # 使用SDK创建声音
            voice = self.client.voices.add(
                name=name,
                files=audio_files,
                description=description,
                labels=labels
            )

            return voice.voice_id

        except Exception as e:
            logger.error(f"添加ElevenLabs声音失败: {str(e)}")
            raise ExternalAPIError(f"添加ElevenLabs声音失败: {str(e)}")

    async def text_to_speech(
            self,
            voice_id: str,
            text: str,
            model_id: str = "eleven_multilingual_v2"
    ) -> str:
        """
        将文本转换为语音并保存为文件

        Args:
            voice_id: 要使用的声音ID
            text: 要转换的文本
            model_id: 模型ID

        Returns:
            保存的音频文件URL（本地路径）
        """
        try:
            # 生成唯一的文件名
            file_name = f"{uuid.uuid4()}.mp3"
            file_path = os.path.join(self.output_dir, file_name)

            # 获取音频内容
            audio_data = self.client.text_to_speech.convert(
                voice_id=voice_id,
                output_format="mp3_44100_128",
                text=text,
                model_id=model_id,
            )

            # 将音频内容保存为文件
            with open(file_path, 'wb') as f:
                # 检查是否为生成器或迭代器
                if hasattr(audio_data, '__iter__') and not isinstance(audio_data, (bytes, bytearray)):
                    # 处理生成器/迭代器情况
                    for chunk in audio_data:
                        f.write(chunk)
                else:
                    # 处理字节数据情况
                    f.write(audio_data)

            # 返回文件URL（本地路径）
            file_url = f"file://{os.path.abspath(file_path)}"
            logger.info(f"保存音频文件成功: {file_url}")
            return file_url

        except Exception as e:
            logger.error(f"ElevenLabs语音生成失败: {str(e)}")
            raise ExternalAPIError(f"ElevenLabs语音生成失败: {str(e)}")

    async def text_to_speech_stream(
            self,
            voice_id: str,
            text: str,
            model_id: str = "eleven_multilingual_v2"
    ) -> str:
        """
        将文本转换为语音流并保存为文件

        Args:
            voice_id: 要使用的声音ID
            text: 要转换的文本
            model_id: 模型ID

        Returns:
            保存的音频文件URL（本地路径）
        """
        try:
            # 生成唯一的文件名
            file_name = f"{uuid.uuid4()}.mp3"
            file_path = os.path.join(self.output_dir, file_name)

            # 获取流式音频
            audio_stream = self.client.text_to_speech.convert_as_stream(
                voice_id=voice_id,
                output_format="mp3_44100_128",
                text=text,
                model_id=model_id,
            )

            # 将流保存为文件
            with open(file_path, 'wb') as f:
                for chunk in audio_stream:
                    f.write(chunk)

            # 返回文件URL（本地路径）
            file_url = f"file://{os.path.abspath(file_path)}"
            logger.info(f"保存音频文件成功: {file_url}")
            return file_url

        except Exception as e:
            logger.error(f"ElevenLabs语音生成失败: {str(e)}")
            raise ExternalAPIError(f"ElevenLabs语音生成失败: {str(e)}")


async def main():
    """测试函数"""
    # 初始化ElevenLabs客户端
    elevenlabs = ElevenLabsClient()

    # 测试 text_to_speech 方法
    file_url1 = await elevenlabs.text_to_speech(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text="This is a test of the ElevenLabs text to speech API."
    )
    print(f"Non-streaming audio saved to: {file_url1}")

    # 测试 create voice 方法
    voice_id = await elevenlabs.add_voice(
        name="Test Voice",
        files=["recording.mp3"],
        description="This is a test voice",
    )
    print(f"New voice created with ID: {voice_id}")


if __name__ == "__main__":
    asyncio.run(main())