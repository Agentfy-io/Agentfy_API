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
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
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

        # 初始化视频数据收集器和清理器
        self.video_collector = VideoCollector(tikhub_api_key)
        self.video_cleaner = VideoCleaner()

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

    def convert_markdown_to_html(self, markdown_content: str, title: str = "Analysis Report") -> str:
        """
        将Markdown内容转换为HTML

        Args:
            markdown_content (str): Markdown内容
            title (str): HTML页面标题

        Returns:
            str: HTML内容
        """
        try:
            import markdown
        except ImportError:
            print("请安装markdown库: pip install markdown")
            return f"<pre>{markdown_content}</pre>"

        # 转换Markdown为HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['tables', 'fenced_code', 'codehilite', 'toc']
        )

        # 创建完整HTML文档
        css = """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }
        h1, h2, h3 { margin-top: 1.5em; color: #111; }
        pre { background-color: #f6f8fa; border-radius: 3px; padding: 16px; overflow: auto; }
        code { font-family: SFMono-Regular, Consolas, Menlo, monospace; background-color: #f6f8fa; padding: 0.2em 0.4em; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f6f8fa; }
        """

        html_document = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>{css}</style>
    </head>
    <body>
        <h1>{title}</h1>
        {html_content}
    </body>
    </html>
        """

        return html_document

    async def fetch_video_data(self, aweme_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        获取指定视频清理后的数据

        Args:
            aweme_id (str): 视频ID

        Returns:
            AsyncGenerator[Dict[str, Any], None]: 异步生成器，产生视频数据
        """
        start_time = time.time()

        try:
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            # 初始状态信息
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': f"开始获取视频数据: {aweme_id}...",
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            logger.info(f"正在获取视频数据: {aweme_id}...")

            video_data = await self.video_collector.collect_single_video(aweme_id)
            cleaned_video_data = await self.video_cleaner.clean_single_video(video_data['video'])
            cleaned_video_data = cleaned_video_data['video']

            # 返回最终结果
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': f"已获取并筛选出关键视频数据: {aweme_id}",
                'video': cleaned_video_data,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            logger.info(f"已获取视频数据: {aweme_id}")

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"获取视频时出错: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': f"获取视频时出错: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

        except Exception as e:
            logger.error(f"获取视频数据时发生未预期错误: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': f"获取视频数据时发生未预期错误: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

    async def analyze_video_info(self, aweme_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        分析视频基础信息

        Args:
            aweme_id (str): 视频ID

        Returns:
            AsyncGenerator[Dict[str, Any], None]: 异步生成器，产生分析结果
        """
        if not aweme_id or not isinstance(aweme_id, str):
            raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

        start_time = time.time()
        llm_processing_cost = 0

        try:
            # 初始状态信息
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': f"开始分析视频基础信息: {aweme_id}...",
                'llm_processing_cost': llm_processing_cost,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            logger.info(f"📊 正在分析视频基础信息: {aweme_id}...")

            # 获取视频数据
            video_data = await self.video_collector.collect_single_video(aweme_id)
            cleaned_video_data = await self.video_cleaner.clean_single_video(video_data['video'])
            cleaned_video_data = cleaned_video_data['video']

            # 调用AI进行分析
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': "正在使用AI分析视频信息...",
                'llm_processing_cost': llm_processing_cost,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            sys_prompt = self.prompts['video_info']
            user_prompt = f"Here is the video data for aweme_id: {aweme_id}\n{cleaned_video_data}"

            # 调用 AI 进行分析
            response = await self.chatgpt.chat(
                system_prompt=sys_prompt,
                user_prompt=user_prompt
            )

            # 解析 AI 返回的结果
            report = response['response']["choices"][0]["message"]["content"].strip()
            llm_processing_cost = response['cost']
            logger.info("已完成视频基础信息分析")

            # 生成报告时更新状态
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': "AI分析完成，正在生成报告...",
                'llm_processing_cost': llm_processing_cost,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            # 保存Markdown报告
            report_dir = "reports"
            os.makedirs(report_dir, exist_ok=True)

            report_md_path = os.path.join(report_dir, f"report_{aweme_id}.md")
            with open(report_md_path, "w", encoding="utf-8") as f:
                f.write(report)

            # 转换为HTML
            html_content = self.convert_markdown_to_html(report, f"video_info Analysis for {aweme_id}")
            html_filename = f"report_{aweme_id}.html"
            html_path = os.path.join(report_dir, html_filename)

            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # 生成本地文件URL
            absolute_path = os.path.abspath(html_path)

            # 构建file://协议URL
            file_url = f"file://{absolute_path}"

            # 确保路径分隔符是URL兼容的
            if os.name == 'nt':  # Windows系统
                # Windows路径需要转换为URL格式
                file_url = file_url.replace('\\', '/')

            logger.info(f"报告已生成: Markdown ({report_md_path}), HTML ({html_path})")
            logger.info(f"报告本地URL: {file_url}")

            # 返回最终结果
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': "视频基础信息分析完成",
                'report': file_url,
                'llm_processing_cost': llm_processing_cost,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"分析视频基础信息时出错: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': f"分析视频基础信息时出错: {str(e)}",
                'error': str(e),
                'llm_processing_cost': llm_processing_cost,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

        except Exception as e:
            logger.error(f"分析视频基础信息时发生未预期错误: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': f"分析视频基础信息时发生未预期错误: {str(e)}",
                'error': str(e),
                'llm_processing_cost': llm_processing_cost,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

    async def fetch_video_transcript(self, aweme_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        分析视频文本转录内容

        Args:
            aweme_id (str): 视频ID

        Returns:
            AsyncGenerator[Dict[str, Any], None]: 异步生成器，产生转录结果
        """
        start_time = time.time()

        try:
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            # 初始状态信息
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': f"正在分析视频文本转录: {aweme_id}...",
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            logger.info(f"正在分析视频文本转录: {aweme_id}...")

            # 获取视频数据
            video_data = None
            async for result in self.fetch_video_data(aweme_id=aweme_id):
                if result['is_complete']:
                    video_data = result.get('video', {})

                    # 传递进度更新
                    yield {
                        'aweme_id': aweme_id,
                        'is_complete': False,
                        'message': "已获取视频数据，准备提取文本转录...",
                        'timestamp': datetime.now().isoformat(),
                        'processing_time_ms': round((time.time() - start_time) * 1000, 2)
                    }

            # 提取视频播放地址
            play_address = video_data.get('play_address', '')

            # 更新状态为正在提取文本
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': "正在提取视频音频文本...",
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
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

            # 返回最终结果
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': "视频文本转录完成",
                'transcript': text,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"分析视频文本转录时出错: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': f"分析视频文本转录时出错: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

        except Exception as e:
            logger.error(f"分析视频文本转录时发生未预期错误: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': f"分析视频文本转录时发生未预期错误: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

    async def analyze_video_frames(self, aweme_id: str, time_interval: float) -> AsyncGenerator[Dict[str, Any], None]:
        """
        分析视频帧内容

        Args:
            aweme_id (str): 视频ID
            time_interval (float): 分析帧的间隔

        Returns:
            AsyncGenerator[Dict[str, Any], None]: 异步生成器，产生分析结果
        """
        start_time = time.time()

        try:
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            # 初始状态信息
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': f"正在分析视频帧内容: {aweme_id}...",
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            logger.info(f"正在分析视频帧内容: {aweme_id}...")

            # 获取视频数据
            video_data = None
            async for result in self.fetch_video_data(aweme_id=aweme_id):
                if result['is_complete']:
                    video_data = result.get('video', {})

                    # 传递进度更新
                    yield {
                        'aweme_id': aweme_id,
                        'is_complete': False,
                        'message': "已获取视频数据，准备分析视频帧...",
                        'timestamp': datetime.now().isoformat(),
                        'processing_time_ms': round((time.time() - start_time) * 1000, 2)
                    }

            # 提取视频播放地址
            play_address = video_data.get('play_address', '')

            # 更新状态为正在分析视频帧
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': f"正在以 {time_interval} 秒间隔分析视频帧...",
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            # 调用 AI 进行分析
            opencv = OpenCV()
            video_script = await opencv.analyze_video(play_address, time_interval)

            # 返回最终结果
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': "视频帧分析完成",
                'video_script': video_script,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"分析视频帧内容时出错: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': f"分析视频帧内容时出错: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

        except Exception as e:
            logger.error(f"分析视频帧内容时发生未预期错误: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': f"分析视频帧内容时发生未预期错误: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

    async def fetch_invideo_text(self, aweme_id: str, time_interval: int = 3, confidence_threshold: float = 0.5) -> \
    AsyncGenerator[Dict[str, Any], None]:
        """
        分析视频中出现的文本内容

        Args:
            aweme_id (str): 视频ID
            time_interval (int): 分析帧的间隔
            confidence_threshold (float): 文本识别的置信度阈值

        Returns:
            AsyncGenerator[Dict[str, Any], None]: 异步生成器，产生提取结果
        """
        start_time = time.time()

        try:
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            # 初始状态信息
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': f"正在分析视频中出现文本内容: {aweme_id}...",
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            logger.info(f"正在分析视频中出现文本内容: {aweme_id}...")

            # 获取视频数据
            video_data = None
            async for result in self.fetch_video_data(aweme_id=aweme_id):
                if result['is_complete']:
                    video_data = result.get('video', {})

                    # 传递进度更新
                    yield {
                        'aweme_id': aweme_id,
                        'is_complete': False,
                        'message': "已获取视频数据，准备提取视频内文本...",
                        'timestamp': datetime.now().isoformat(),
                        'processing_time_ms': round((time.time() - start_time) * 1000, 2)
                    }
            # 提取视频播放地址
            play_address = video_data.get('play_address', '')

            # 更新状态为正在提取文本
            yield {
                'aweme_id': aweme_id,
                'is_complete': False,
                'message': f"正在以 {time_interval} 秒间隔提取视频内文本，置信度阈值：{confidence_threshold}...",
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            # 调用 AI 进行分析
            video_ocr = VideoOCR()
            # 提取视频中的文本内容
            texts = await video_ocr.analyze_video(play_address, time_interval, confidence_threshold)

            # 返回最终结果
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': "视频内文本提取完成",
                'in_video_texts': texts,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"分析视频文本内容时出错: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': f"分析视频文本内容时出错: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

        except Exception as e:
            logger.error(f"分析视频文本内容时发生未预期错误: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'message': f"分析视频文本内容时发生未预期错误: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }


