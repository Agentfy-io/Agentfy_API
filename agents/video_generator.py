# -*- coding: utf-8 -*-
"""
@file: video_generator.py
@desc: 根据用户不同需求生成视频内容·
@auth: Callmeiks
"""

import json
import re
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, AsyncGenerator

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from app.utils.logger import setup_logger
from services.ai_models.chatgpt import ChatGPT
from services.ai_models.claude import Claude
from services.ai_models.whisper import WhisperLemonFox
from services.ai_models.genny import Genny
from services.ai_models.elevenLabs import ElevenLabsClient
from app.config import settings
from app.core.exceptions import ValidationError, ExternalAPIError, InternalServerError
from services.auth.auth_service import user_sessions
from agents.audio_generator import AudioGeneratorAgent


# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()


class VideoGeneratorAgent:
    """
    为创作者生成视频内容的代理类
    """

    def __init__(self, tikhub_api_key: Optional[str] = None):
        """
        初始化音频生成代理类

        Args:
            tikhub_api_key: TikHub API密钥
        """
        # 初始化AI模型客户端
        self.chatgpt = ChatGPT()
        self.claude = Claude()
        self.whisper = WhisperLemonFox()
        self.genny = Genny()
        self.elevenLabs = ElevenLabsClient()

        # 初始化内置agent
        self.audio_generator = AudioGeneratorAgent()

        # 保存TikHub API配置
        self.tikhub_api_key = tikhub_api_key
        self.tikhub_base_url = settings.TIKHUB_BASE_URL

        # 如果没有提供TikHub API密钥，记录警告
        if not self.tikhub_api_key:
            logger.warning("未提供TikHub API密钥，某些功能可能不可用")

        # 加载系统和用户提示
        self._load_system_prompts()

    def _load_system_prompts(self) -> None:
        """加载系统提示用于不同的评论分析类型"""
        self.system_prompts = {
            "video_segments": """You are a professional livestream editor specializing in extracting the most engaging clips for TikTok's young audience from long recordings.          
                INPUT:
                1. Complete livestream transcript with timestamps (Multilingual)
                2. Number of segments to extract (N)
                
                TASK:
                - Analyze the entire transcript to identify the most interesting and engaging moments
                - Focus specifically on these types of content:
                  * Gossip or exclusive revelations
                  * Host's emotional reactions or intense moments
                  * Important product introductions or recommendations
                  * Surprising or controversial opinions
                  * Humorous interactions or stories
                - Select N segments with the highest viral potential
                - Keep each segment under 1 minute (ideal length: 30-45 seconds)
                - Ensure selected segments have coherent content with clear beginning and ending points
                - Avoid segments with empty content or long silent periods
                - IMPORTANT: The selected segments do NOT need to cover the entire video content - you are only selecting the most interesting moments, and large portions of the original video may be excluded
                
                OUTPUT FORMAT:
                Return a JSON array of objects, where each object contains:
                1. segment_number: Sequential number of the segment
                2. start_time: Timestamp where the segment starts in "HH:MM:SS" format
                3. end_time: Timestamp where the segment ends in "HH:MM:SS" format
                4. duration_seconds: Length of the segment in seconds
                5. content: The transcript text for this segment
                6. highlight_reason: Brief explanation of why this segment would appeal to young viewers (e.g., gossip, product reveal, emotional peak)
                7. suggested_title: Recommended clickbait title for the short video
                
                IMPORTANT GUIDELINES:
                - Prioritize content-dense segments with good pacing
                - Input transcript may be multilingual, output segments should be the same language as the original.
                - Select segments that can be understood without additional context
                - Avoid content requiring extensive background knowledge
                - Choose segments with attention-grabbing first few seconds
                - Minimize content overlap between selected segments
                - It's perfectly fine if your selected segments only cover a small percentage of the total video length - quality matters more than coverage
                
                Example output:
                ```json
                {
                  "segments": [
                    {
                      "segment_number": 1,
                      "start_time": "00:15:23",
                      "end_time": "00:16:12",
                      "duration_seconds": 49,
                      "content": "Actually, I met with Musk last week and he told me they're developing a completely new AI product that hasn't been announced yet...",
                      "highlight_reason": "Exclusive revelation about unreleased product information that will spark discussion",
                      "suggested_title": "SHOCKING! Streamer reveals Musk's secret AI plans #TechLeaks"
                    },
                    ...
                  ],
                  "total_segments": N,
                  "total_duration_seconds": 293
                }
            """
        }

    # TODO: 读取长视频的音频文件，变成transcript 包括每句话对应的时间戳
    async def get_transcript(self, file: str) -> Dict:
        """
        获取音频文件的转录文本

        Args:
            file: 音频文件路径

        Returns:
            转录文本
        """
        try:
            # 从AI模型获取音频转录
            transcript = await self.whisper.transcriptions(file = file, timestamp_granularities= ["sentence"])
            logger.info(f"获取音频转录成功: {transcript}")
            return transcript
        except Exception as e:
            logger.error(f"获取音频转录失败: {e}")
            raise InternalServerError("获取音频转录失败")

    # TODO： 根据transcript以及用户自定义的切片量选出视频片段们，然后将这些片段的时间戳返回
    async def get_video_segments(self, transcript: Dict, segment_count: int = 5) -> List[Dict]:
        """
        根据音频转录文本获取视频片段时间戳

        Args:
            transcript: 音频转录文本
            segment_length: 视频片段长度

        Returns:
            视频片段时间戳
        """
        try:
            sys_prompt = self.system_prompts["video_segments"]
            user_prompt = f"Here's full transcript: {transcript}, and I want {segment_count} segments"

            # 从AI模型获取视频片段时间戳
            result = await self.chatgpt.chat(sys_prompt, user_prompt)

            result = result['response']["choices"][0]["message"]["content"].strip()
            result = json.loads(result)

            video_segments = result["segments"]

            logger.info(f"获取视频片段时间戳成功: {video_segments}")
            return video_segments
        except Exception as e:
            logger.error(f"获取视频片段时间戳失败: {e}")
            raise InternalServerError("获取视频片段时间戳失败")

    # TODO： 根据选出片段的时间戳，将原视频切割成小片段，然后每个小片段单独保存为file ，最后只需要返回这些小片段的file路径list即可
    async def generate_video_segments(self, original_file: str, segments: List[Dict]) -> List[str]:
        """
        生成视频片段

        Args:
            original_file: 原始视频文件路径
            segments: 视频片段时间戳

        Returns:
            视频片段文件路径列表
        """
        # 生成视频片段文件路径列表
        segment_files = []
        # 生成视频片段
        for i, segment in enumerate(segments):
            try:
                start_time = segment["start_time"]
                end_time = segment["end_time"]
                duration = segment["duration_seconds"]

                # 生成视频片段文件路径
                segment_file = f"{original_file}_{i}.mp4"
                segment_files.append(segment_file)

                # 使用ffmpeg切割视频片段
                os.system(f"ffmpeg -i {original_file} -ss {start_time} -t {duration} -c copy {segment_file}")

                logger.info(f"生成视频片段成功: {segment_file}")
            except Exception as e:
                logger.error(f"片段{i}生成失败: {e}，跳过")
                continue
        return segment_files

    async def generate_video(self, file: str, segment_count: int = 5) -> AsyncGenerator[Dict[str, Any], None]:
        """
        生成视频
        Args:
            file:
            segment_count:

        Returns:

        """
        start_time = time.time()
        try:
            # 获取音频转录
            transcript = await self.get_transcript(file)
            yield {
                "message": "音频转录成功",
                "transcript": transcript
            }
            # 获取视频片段时间戳
            segments = await self.get_video_segments(transcript, segment_count)
            yield {
                "message": "获取视频片段时间戳成功",
                "segments": segments
            }
            # 生成视频片段
            segment_files = await self.generate_video_segments(file, segments)
            # 生成视频
            for segment_file in segment_files:
                yield {"file": segment_file}
        except Exception as e:
            logger.error(f"生成视频失败: {e}")
            raise InternalServerError("生成视频失败")
        finally:
            logger.info(f"生成视频总耗时: {time.time() - start_time:.2f}秒")

async def main():
    video_generator = VideoGeneratorAgent()
    async for video in video_generator.generate_video("test.mp4", 5):
        print(video)


if __name__ == "__main__":
    asyncio.run(main())



