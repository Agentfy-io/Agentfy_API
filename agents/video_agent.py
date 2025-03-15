# -*- coding: utf-8 -*-
"""
@file: video_agent.py
@desc: 视频分析器，用于分析TikTok视频数据并生成综合报告
@auth: Callmeiks
"""

import json
import re
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import asyncio
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

from app.utils.logger import setup_logger
from services.ai_models.chatgpt import ChatGPT
from services.ai_models.claude import Claude
from services.ai_models.opencv import OpenCV
from services.ai_models.videoOCR import VideoOCR
from services.ai_models.whisper import WhisperLemonFox
from services.cleaner.comment_cleaner import CommentCleaner
from services.crawler.comment_crawler import CommentCollector
from services.crawler.video_crawler import VideoCollector
from services.cleaner.video_cleaner import VideoCleaner
from app.config import settings
from markdown import markdown  # pip install markdown
from app.core.exceptions import ValidationError, ExternalAPIError, InternalServerError

# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()


class VideoAgent:
    """视频全方位分析器，用于分析TikTok视频数据并生成综合报告。"""

    def __init__(self, tikhub_api_key: Optional[str] = None):
        """
        初始化VideoAgent。

        Args:
            tikhub_api_key: TikHub API密钥
            tikhub_base_url: TikHub API基础URL
        """
        # 初始化AI模型客户端
        self.chatgpt = ChatGPT()
        self.claude = Claude()

        # 保存TikHub API配置
        self.tikhub_api_key = tikhub_api_key
        self.tikhub_base_url = settings.TIKHUB_BASE_URL

        # 如果没有提供TikHub API密钥，记录警告
        if not self.tikhub_api_key:
            logger.warning("未提供TikHub API密钥，某些功能可能不可用")

        # 支持的分析类型列表
        self.analysis_types = ['purchase_intent']

        # 加载系统和用户提示
        self._load_system_prompts()
        # self._load_user_prompts()

    def _load_system_prompts(self):
        self.prompts = {
            "video_info": """You are a social media video analyst. Your task is to analyze a TikTok video's data and create a comprehensive report focusing on key metrics and insights.
                Please analyze the video data and create a report with the following sections:

                1. Basic Video Information
                | Metric | Value |
                | --- | --- |
                | Video ID (aweme_id) | ... |
                | Creation Time | (convert timestamp to readable date) |
                | Description | ... |
                | Duration | (convert to seconds/minutes format) |
                | Content Type | ... |
                | Group ID | ... |
                | Region | ... |

                2. Creator Information
                | Creator Info | Details |
                | --- | --- |
                | Username (unique_id) | ... |
                | Nickname | ... |
                | User ID (uid) | ... |
                | Region | ... |
                | Avatar URL | ... |
                | YouTube Channel | (combine id and title) |
                | Instagram ID | ... |
                | Twitter Info | (if available) |

                3. Music Information
                | Music Details | Value |
                | --- | --- |
                | Music ID | ... |
                | Title | ... |
                | Owner | (combine owner_id and owner_nickname) |
                | Play URL | ... |

                4. Engagement Statistics
                | Metric | Count | Rate (% of views) |
                | --- | --- | --- |
                | Views (play_count) | ... | 100% |
                | Likes (digg_count) | ... | ...% |
                | Comments | ... | ...% |
                | Shares | ... | ...% |
                | Collections (collect_count) | ... | ...% |
                | Downloads | ... | ...% |

                5. Content Features
                | Feature | Status |
                | --- | --- |
                | AI Generated | (created_by_ai) |
                | Challenge List | (cha_list details) |
                | Hashtags | (list all) |
                | Is Top Content | (is_top) |
                | VR Content | (is_vr) |
                | PGC Show | (is_pgcshow) |
                | Advertisement | (is_ads) |
                | VS Entry | (has_vs_entry) |
                | Feed Eligible | (is_nff_or_nr) |

                6. Interactive Features
                | Feature | Status |
                | --- | --- |
                | Duet Allowed | (item_duet) |
                | React Allowed | (item_react) |
                | Stitch Allowed | (item_stitch) |
                | Download Allowed | (allow_download) |

                7. Content Status
                | Status Check | Value |
                | --- | --- |
                | Under Review | (is_reviewing) |
                | Prohibited | (is_prohibited) |
                | Deleted | (is_delete) |
                | Review Status | (reviewed) |

                8. Sharing Information
                | Share Details | Value |
                | --- | --- |
                | Share URL | ... |
                | Share Description | ... |
                | Download Address | ... |

                9. Key Performance Indicators (Calculated)
                - Engagement Rate: (likes + comments + shares) / views * 100
                - Collection Rate: collections / views * 100
                - Share Rate: shares / views * 100
                - Download Rate: downloads / views * 100
                - Overall Interaction Rate: (all interactions) / views * 100
                - summary of all data 

                Please format all numbers appropriately:
                - Use K for thousands (e.g., 1.5K)
                - Use M for millions (e.g., 2.3M)
                - Use B for billions
                - Round percentages to 2 decimal places

                End the report with:
                1. A summary of the video's overall performance
                2. Notable strengths
                3. Areas for potential improvement
                4. Content distribution recommendations based on all metrics """,
        }
        # 定义函数映射
        self.function_map = {
            "video_info": self.analyze_video_info,
            "video_transcript": self.fetch_video_transcript,
            "video_frames": self.analyze_video_frames,
            "in_video_text": self.fetch_invideo_text
        }

    async def fetch_video_data(self, aweme_id: str) -> Dict[str, Any]:
        """
        获取指定视频清理后的数据

        Args:
            aweme_id (str): 视频ID

        Returns:
            Dict[str, Any]: 视频数据
        """

        start_time = time.time()

        try:
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            logger.info(f"🔍 正在获取视频数据: {aweme_id}...")

            video_crawler = VideoCollector(self.tikhub_api_key)
            video_data = await video_crawler.collect_single_video(aweme_id)

            if not video_data.get('video'):
                logger.warning(f"❌ 未找到视频数据: {aweme_id}")
                return{
                    'aweme_id': aweme_id,
                    'video': None,
                    'timestamp': datetime.now().isoformat()
                }
            video_cleaner = VideoCleaner()
            cleaned_video_data = await video_cleaner.clean_single_video(video_data['video'])
            cleaned_video_data = cleaned_video_data['video']

            result = {
                'aweme_id': aweme_id,
                'video': cleaned_video_data,
                'timestamp': datetime.now().isoformat(),
                'processing_time': round(time.time() - start_time, 2)
            }

            logger.info(f"✅ 已获取视频数据: {aweme_id}")
            return result

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"获取视频时出错: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"获取视频数据时发生未预期错误: {str(e)}")
            raise InternalServerError(detail=f"获取视频数据时发生未预期错误: {str(e)}")


    async def analyze_video_info(self, aweme_id: str) -> Dict[str, Any]:
        """
        分析视频基础信息

        Args:
            aweme_id (str): 视频ID

        Returns:
            Dict[str, Any]: 分析结果
        """

        start_time = time.time()

        try:
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            logger.info(f"📊 正在分析视频基础信息: {aweme_id}...")

            video_data = await self.fetch_video_data(aweme_id=aweme_id)
            data = video_data.get('video', {})

            if not data:
                logger.warning(f"❌ 未找到视频数据: {aweme_id}")
                return {
                    'aweme_id': aweme_id,
                    'video_info': None,
                    'timestamp': datetime.now().isoformat()
                }

            sys_prompt = self.prompts['video_info']
            user_prompt = f"Here is the video data for aweme_id: {aweme_id}\n{data}"
            # 调用 AI 进行分析
            response = await self.chatgpt.chat(
                system_prompt=sys_prompt,
                user_prompt=user_prompt
            )

            # 解析 AI 返回的结果
            analysis_results = response["choices"][0]["message"]["content"].strip()
            logger.info("✅ 已完成用户/达人基础信息分析")

            # 将 Markdown 转换为 HTML
            analysis_html = markdown(analysis_results)

            # 生成一个唯一文件名
            unique_id = str(uuid.uuid4())
            file_name = f"report_{unique_id}.html"

            # 将 HTML 写入本地文件
            file_path = os.path.join("./reports", file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(analysis_html)

            # 这里返回的 temp_display_url 就是本地的文件路径或相对路径
            # 具体如何对外访问，需要看你如何配置路由或静态文件服务
            temp_display_url = file_path

            logger.info("✅ 已完成用户/达人基础信息分析")

            return {
                'aweme_id': aweme_id,
                'video_info_html': analysis_html,  # 转换后的 HTML
                'temp_display_url': temp_display_url,  # 存储的文件路径
                'timestamp': datetime.now().isoformat(),
                'processing_time': round(time.time() - start_time, 2)
            }

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"❌ 分析视频基础信息时出错: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"❌ 分析视频基础信息时发生未预期错误: {str(e)}")
            raise InternalServerError(detail=f"分析视频基础信息时发生未预期错误: {str(e)}")

    async def fetch_video_transcript(self, aweme_id: str) -> Dict[str, Any]:
        """
        分析视频文本转录内容

        Args:
            aweme_id (str): 视频ID

        Returns:
            Dict[str, Any]: 分析结果
        """
        start_time = time.time()

        try:
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            logger.info(f"🔍 正在分析视频文本转录: {aweme_id}...")

            video_data = await self.fetch_video_data(aweme_id=aweme_id)
            data = video_data.get('video', {})

            if not data:
                logger.warning(f"❌ 未找到视频数据: {aweme_id}")
                return {
                    'aweme_id': aweme_id,
                    'transcript': None,
                    'timestamp': datetime.now().isoformat()
                }

            # 提取视频播放地址
            play_address = data.get('play_address', '')
            if not play_address:
                logger.warning(f"❌ 未找到视频播放地址: {aweme_id}")
                return {
                    'aweme_id': aweme_id,
                    'transcript': None,
                    'timestamp': datetime.now().isoformat()
                }

            # 调用 AI 进行分析
            whisper = WhisperLemonFox()

            # 获取视频文本转录
            transcript = await whisper.transcriptions(
                file=play_address,
                response_format="verbose_json",
                speaker_labels=False,
                prompt="",
                language="",
                callback_url="",
                translate=False,
                timestamp_granularities=None,
                timeout=60
            )
            # 提取文本内容
            text = transcript.get('text', '')
            return {
                'aweme_id': aweme_id,
                'transcript': text,
                'timestamp': datetime.now().isoformat(),
                'processing_time': round(time.time() - start_time, 2)
            }

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"❌ 分析视频文本转录时出错: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"❌ 分析视频文本转录时发生未预期错误: {str(e)}")
            raise InternalServerError(detail=f"分析视频文本转录时发生未预期错误: {str(e)}")

    async def analyze_video_frames(self, aweme_id: str, time_interval: float) -> Dict[str, Any]:
        """
        分析视频帧内容

        Args:
            aweme_id (str): 视频ID
            time_interval (float): 分析帧的间隔

        Returns:
            Dict[str, Any]: 分析结果
        """

        start_time = time.time()

        try:
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            logger.info(f"🔍 正在分析视频帧内容: {aweme_id}...")

            video_data = await self.fetch_video_data(aweme_id=aweme_id)
            data = video_data.get('video', {})

            if not data:
                logger.warning(f"❌ 未找到视频数据: {aweme_id}")
                return {
                    'aweme_id': aweme_id,
                    'video_script': None,
                    'timestamp': datetime.now().isoformat()
                }

            # 提取视频播放地址
            play_address = data.get('play_address', '')
            if not play_address:
                logger.warning(f"❌ 未找到视频播放地址: {aweme_id}")
                return {
                    'aweme_id': aweme_id,
                    'video_script': None,
                    'timestamp': datetime.now().isoformat()
                }

            # 调用 AI 进行分析
            opencv = OpenCV()
            video_script = await opencv.analyze_video(play_address, time_interval)

            return {
                'aweme_id': aweme_id,
                'video_script': video_script,
                'timestamp': datetime.now().isoformat(),
                'processing_time': round(time.time() - start_time, 2)
            }

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"❌ 分析视频帧内容时出错: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"❌ 分析视频帧内容时发生未预期错误: {str(e)}")
            raise InternalServerError(detail=f"分析视频帧内容时发生未预期错误: {str(e)}")

    async def fetch_invideo_text(self, aweme_id: str, time_interval: int = 3, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        分析视频中出现的文本内容

        Args:
            aweme_id (str): 视频ID
            time_interval (int): 分析帧的间隔
            confidence_threshold (float): 文本识别的置信度阈值

        Returns:
            Dict[str, Any]: 分析结果
        """

        start_time = time.time()

        try:
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            logger.info(f"🔍 正在分析视频中出现文本内容: {aweme_id}...")

            video_data = await self.fetch_video_data(aweme_id=aweme_id)
            data = video_data.get('video', {})

            if not data:
                logger.warning(f"❌ 未找到视频数据: {aweme_id}")
                return {
                    'aweme_id': aweme_id,
                    'texts': None,
                    'timestamp': datetime.now().isoformat()
                }

            # 提取视频播放地址
            play_address = data.get('play_address', '')
            if not play_address:
                logger.warning(f"❌ 未找到视频播放地址: {aweme_id}")
                return {
                    'aweme_id': aweme_id,
                    'texts': None,
                    'timestamp': datetime.now().isoformat()
                }

            # 调用 AI 进行分析
            video_ocr = VideoOCR()
            # 提取视频中的文本内容
            texts = await video_ocr.analyze_video(play_address, time_interval, confidence_threshold)

            return {
                'aweme_id': aweme_id,
                'in_video_texts': texts,
                'timestamp': datetime.now().isoformat(),
                'processing_time': round(time.time() - start_time, 2)
            }

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"❌ 分析视频文本内容时出错: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"❌ 分析视频文本内容时发生未预期错误: {str(e)}")
            raise InternalServerError(detail=f"分析视频文本内容时发生未预期错误: {str(e)}")



