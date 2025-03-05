# -*- coding: utf-8 -*-
"""
@file: sentiment_agent.py
@desc: 处理TikTok评论的代理类，提供评论获取、舆情分析和黑粉识别功能
@auth: Callmeiks
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from collections import Counter
import asyncio
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

from app.utils.logger import setup_logger
from services.ai_models.chatgpt import ChatGPT
from services.ai_models.claude import Claude
from services.cleaner.comment_cleaner import CommentCleaner
from services.crawler.comment_crawler import CommentCollector
from services.crawler.video_crawler import VideoCollector
from services.cleaner.video_cleaner import VideoCleaner
from app.config import settings
from app.core.exceptions import ValidationError, ExternalAPIError, InternalServerError

# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()


class SentimentAgent:
    """处理TikTok评论的代理类，提供评论获取、舆情分析和黑粉识别功能"""

    def __init__(self, tikhub_api_key: Optional[str] = None, tikhub_base_url: Optional[str] = None):
        """
        初始化SentimentAgent，加载API密钥和提示模板

        Args:
            tikhub_api_key: TikHub API密钥
            tikhub_base_url: TikHub API基础URL
        """
        # 初始化AI模型客户端
        self.chatgpt = ChatGPT()
        self.claude = Claude()

        # 保存TikHub API配置
        self.tikhub_api_key = tikhub_api_key
        self.tikhub_base_url = tikhub_base_url

        # 如果没有提供TikHub API密钥，记录警告
        if not self.tikhub_api_key:
            logger.warning("未提供TikHub API密钥，某些功能可能不可用")

        # 加载系统和用户提示
        self._load_system_prompts()
        self._load_user_prompts()

    def _load_system_prompts(self) -> None:
        """加载系统提示用于不同的评论分析类型"""
        self.system_prompts = {
            'sentiment': """You are an AI trained to analyze social media comments on TikTok videos. Your task is to categorize and summarize how commenters react to a given video.
                For each comment in the provided list, analyze:
                1. Sentiment (positive, negative, or neutral)
                2. Emotion (e.g., excitement, amusement, frustration, curiosity, etc.)
                3. Engagement Type (e.g., supportive, critical, humorous, questioning, spam)
                4. Key Themes (e.g., reaction to video content, reference to trends, personal anecdotes, meme usage)
                5. Virality Indicators (e.g., high engagement phrases, common phrases from viral trends, use of emojis, repeated comment patterns)
                
                Return the analysis in the following JSON format, and comment_id should still remain the same as input data:
                [
                    {
                        "comment_id": "comment ID from input data",
                        "text": "comment text",
                        "sentiment": "positive/negative/neutral",
                        "emotion": "emotion_label",
                        "engagement_type": "supportive/critical/humorous/questioning/spam",
                        "key_themes": ["theme1", "theme2"],
                        "virality_indicators": ["emoji", "phrase", "meme_reference"],
                    }
                ]
                
                Guidelines:
                - Identify emotions based on wording, punctuation, and emojis.
                - Consider trends and viral elements influencing reactions.
                - If a comment uses humor or sarcasm, categorize it accordingly.
                - If a comment references a meme or viral phrase, list it as a key theme.
                - Comments containing excessive emojis, repeated phrases, or engagement-bait (e.g., "like if you agree") should be marked as potential virality indicators.
                
                Respond with a JSON array containing analysis for all comments.
                """,

    def _load_user_prompts(self) -> None:
        """加载用户提示用于不同的评论分析类型"""
        self.user_prompts = {
            'sentiment': {
                'description': "sentiment and engagement analysis",
    async def fetch_video_comments(self, aweme_id: str) -> Dict[str, Any]:
        """
        获取指定视频的清理后的评论数据

        Args:
            aweme_id (str): 视频ID

        Returns:
            Dict[str, Any]: 清理后的评论数据，包含视频ID和评论列表

        Raises:
            ValidationError: 当aweme_id为空或无效时
            ExternalAPIError: 当网络连接失败时
        """
        start_time = time.time()

        try:
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            # 记录开始获取评论
            logger.info(f"开始获取视频 {aweme_id} 的评论")

            # 获取评论
            comment_collector = CommentCollector(self.tikhub_api_key, self.tikhub_base_url)
            comments = await comment_collector.collect_video_comments(aweme_id)

            if not comments or not comments.get('comments'):
                logger.warning(f"视频 {aweme_id} 未找到评论")
                return {
                    'aweme_id': aweme_id,
                    'comments': [],
                    'comment_count': 0,
                    'timestamp': datetime.now().isoformat()
                }

            # 清洗评论
            comment_cleaner = CommentCleaner()
            cleaned_comments = await comment_cleaner.clean_video_comments(comments)
            cleaned_comments = cleaned_comments.get('comments', [])

            processing_time = time.time() - start_time

            result = {
                'aweme_id': aweme_id,
                'comments': cleaned_comments,
                'comment_count': len(cleaned_comments),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round(processing_time * 1000, 2)
            }

            logger.info(f"成功获取视频 {aweme_id} 的评论: {len(cleaned_comments)} 条，耗时: {processing_time:.2f}秒")
            return result

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"获取视频评论时出错: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"获取视频评论时发生未预期错误: {str(e)}")
            raise InternalServerError(detail=f"获取视频评论时发生未预期错误: {str(e)}")

    async def analyze_comments_batch(
            self,
            df: pd.DataFrame,
            analysis_type: str,
            batch_size: int = 30,
            concurrency: int = 5
    ) -> pd.DataFrame:
        """
        分批分析评论，以防止一次性发送过多数据

        Args:
            df (pd.DataFrame): 包含评论数据的DataFrame，需包含字段：text
            analysis_type (str): 选择分析类型 (purchase_intent)
            batch_size (int, optional): 每批处理的评论数量，默认30
            concurrency (int, optional): 并发处理的批次数量，默认5

        Returns:
            pd.DataFrame: 结果合并回原始DataFrame

        Raises:
            ValidationError: 当analysis_type无效或df为空时
            InternalServerError: 当分析过程中出现错误时
        """
        try:
            # 参数验证
            if analysis_type not in self.analysis_types:
                raise ValidationError(
                    detail=f"无效的分析类型: {analysis_type}. 请从 {self.analysis_types} 中选择",
                    field="analysis_type"
                )

            if df.empty:
                raise ValidationError(detail="DataFrame不能为空", field="df")

            if 'text' not in df.columns:
                raise ValidationError(detail="DataFrame必须包含'text'列", field="df")

            # 验证和调整批处理参数
            batch_size = min(batch_size, settings.MAX_BATCH_SIZE)
            concurrency = min(concurrency, 10)  # 限制最大并发数为10

            # 分批处理DataFrame
            num_splits = max(1, len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0))
            comment_batches = np.array_split(df, num_splits)
            logger.info(
                f"🚀 开始 {analysis_type} 分析，共 {len(comment_batches)} 批，每批约 {len(comment_batches[0]) if len(comment_batches) > 0 else 0} 条评论"
            )

            # 并发执行任务（每次最多 `concurrency` 组）
            results = []
            for i in range(0, len(comment_batches), concurrency):
                batch_group = comment_batches[i:i + concurrency]

                # 记录当前并发组的范围
                batch_indices = [
                    f"{batch.index[0]}-{batch.index[-1]}" if not batch.empty else "-"
                    for batch in batch_group
                ]
                logger.info(
                    f"⚡ 处理批次 {i + 1} 至 {i + len(batch_group)}，评论索引范围: {batch_indices}"
                )

                # 并发执行 `concurrency` 个任务
                tasks = [
                    self._analyze_aspect(analysis_type, batch[['text', 'comment_id']].to_dict('records'))
                    for batch in batch_group if not batch.empty
                ]

                if not tasks:
                    continue

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # 处理结果，过滤掉异常
                valid_results = []
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"批次 {i + j + 1} 分析失败: {str(result)}")
                    else:
                        valid_results.append(result)

                if len(valid_results) != len(batch_group):
                    logger.warning(
                        f"批次 {i + 1} 至 {i + len(batch_group)} 中有 {len(batch_group) - len(valid_results)} 个批次分析失败"
                    )

                results.extend(valid_results)

            # 检查是否有结果
            if not results:
                raise InternalServerError("所有批次分析均失败，未获得有效结果")

            # 合并所有分析结果
            try:
                # 将所有结果扁平化为单个列表
                all_results = []
                for batch_result in results:
                    if isinstance(batch_result, list):
                        all_results.extend(batch_result)

                # 创建结果DataFrame
                analysis_df = pd.DataFrame(all_results)

                # 确保comment_id列存在
                if 'comment_id' not in analysis_df.columns:
                    logger.warning("分析结果缺少comment_id列，无法正确合并")
                    # 添加索引作为临时列
                    analysis_df['temp_index'] = range(len(analysis_df))
                    df['temp_index'] = range(len(df))
                    # 基于索引合并
                    merged_df = pd.merge(df, analysis_df, on='temp_index', how='left')
                    merged_df = merged_df.drop('temp_index', axis=1)
                else:
                    # 基于comment_id合并
                    merged_df = pd.merge(df, analysis_df, on='comment_id', how='left')

                # 处理重复的text列
                if 'text_y' in merged_df.columns:
                    merged_df = merged_df.drop('text_y', axis=1)
                    merged_df = merged_df.rename(columns={'text_x': 'text'})

                logger.info(f"✅ {analysis_type} 分析完成！总计 {len(merged_df)} 条数据")
                return merged_df

            except Exception as e:
                logger.error(f"合并分析结果时出错: {str(e)}")
                raise InternalServerError(f"合并分析结果时出错: {str(e)}")

        except ValidationError:
            # 直接向上传递验证错误
            raise
        except InternalServerError:
            # 直接向上传递内部服务器错误
            raise
        except Exception as e:
            logger.error(f"分析评论时发生未预期错误: {str(e)}")
            raise InternalServerError(f"分析评论时发生未预期错误: {str(e)}")

    async def _analyze_aspect(
            self,
            aspect_type: str,
            comment_data: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        通用分析方法，根据不同的分析类型调用ChatGPT或Claude AI模型。

        Args:
            aspect_type (str): 需要分析的类型 (purchase_intent)
            comment_data (List[Dict[str, Any]]): 需要分析的评论列表

        Returns:
            Optional[List[Dict[str, Any]]]: AI返回的分析结果，失败时抛出异常

        Raises:
            ValidationError: 当aspect_type无效时
            ExternalAPIError: 当调用AI服务时出错
        """
        try:
            if aspect_type not in self.analysis_types:
                raise ValidationError(detail=f"不支持的分析类型: {aspect_type}", field="aspect_type")

            if not comment_data:
                logger.warning("评论数据为空，跳过分析")
                return []

            # 获取分析的系统提示和用户提示
            aspect_config = self.user_prompts[aspect_type]
            sys_prompt = self.system_prompts[aspect_type]
            user_prompt = (
                f"Analyze the {aspect_config['description']} for the following comments:\n"
                f"{json.dumps(comment_data, ensure_ascii=False)}"
            )

            # 为避免token限制，限制评论文本长度
            for comment in comment_data:
                if 'text' in comment and len(comment['text']) > 1000:
                    comment['text'] = comment['text'][:997] + "..."

            # 调用AI进行分析，优先使用ChatGPT，失败时尝试Claude
            try:
                response = await self.chatgpt.chat(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt
                )

                # 解析ChatGPT返回的结果
                analysis_results = response["choices"][0]["message"]["content"].strip()

            except ExternalAPIError as e:
                logger.warning(f"ChatGPT分析失败，尝试使用Claude: {str(e)}")
                try:
                    # 尝试使用Claude作为备份
                    response = await self.claude.chat(
                        system_prompt=sys_prompt,
                        user_prompt=user_prompt
                    )
                    analysis_results = response["choices"][0]["message"]["content"].strip()
                except Exception as claude_error:
                    logger.error(f"Claude分析也失败: {str(claude_error)}")
                    raise ExternalAPIError(
                        detail="所有AI服务均无法完成分析",
                        service="AI"
                    )

            # 处理返回的JSON格式（可能包含在Markdown代码块中）
            analysis_results = re.sub(
                r"```json\n|\n```|```|\n",
                "",
                analysis_results.strip()
            )

            try:
                analysis_result = json.loads(analysis_results)
                return analysis_result
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {str(e)}, 原始内容: {analysis_results[:200]}...")
                raise ExternalAPIError(
                    detail="AI返回的结果无法解析为JSON",
                    service="AI",
                    original_error=e
                )

        except ValidationError:
            # 直接向上传递验证错误
            raise
        except ExternalAPIError:
            # 直接向上传递API错误
            raise
        except Exception as e:
            logger.error(f"分析评论方面时发生未预期错误: {str(e)}")
            raise InternalServerError(f"分析评论方面时发生未预期错误: {str(e)}")

    async def analyze_sentiment(self, aweme_id: str, batch_size:int=30,concurrency: int=5) -> Dict[str, Any]:
        """
        分析指定视频的评论情感

        Args:
            aweme_id (str): 视频ID
            batch_size (int, optional): 每批处理的评论数量，默认30
            concurrency (int, optional): 并发处理的批次数量，默认5

        Returns:
            Dict[str, Any]: 情感分析结果

        Raises:
            ValidationError: 当aweme_id为空或无效时
            ExternalAPIError: 当网络连接失败时
            InternalServerError: 当分析过程中出现错误时
        """

        start_time = time.time()

        try:
            if not aweme_id:
                raise ValidationError(detail="aweme_id不能为空", field="aweme_id")

            if batch_size <= 0 or batch_size > settings.MAX_BATCH_SIZE:
                raise ValidationError(
                    detail=f"batch_size必须在1和{settings.MAX_BATCH_SIZE}之间",
                    field="batch_size"
                )

            # 获取清理后的评论数据
            comments_data = await self.fetch_video_comments(aweme_id)

            if not comments_data.get('comments'):
                logger.warning(f"视频 {aweme_id} 没有评论数据")
                return {
                    'aweme_id': aweme_id,
                    'error': 'No comments found',
                    'analysis_timestamp': datetime.now().isoformat(),
                }

            comments_df = pd.DataFrame(comments_data['comments'])

            logger.info(f"开始分析视频 {aweme_id} 的评论情感")
            analyzed_df = await self.analyze_comments_batch(comments_df, 'sentiment', batch_size, concurrency)

            if analyzed_df.empty:
                raise InternalServerError("未获得有效的情感分析结果")
            analysis_summary = {
                'sentiment_distribution': self.analyze_sentiment_distribution(analyzed_df),
                'emotion_patterns': self.analyze_emotion_patterns(analyzed_df),
                'engagement_patterns': self.analyze_engagement_patterns(analyzed_df),
                'themes': self.analyze_themes(analyzed_df),
                'meta': {
                    'total_analyzed_comments': len(analyzed_df),
                    'aweme_id': aweme_id,
                    'analysis_type': 'sentiment',
                    'analysis_timestamp': datetime.now().isoformat(),
                    'processing_time_ms': round((time.time() - start_time) * 1000, 2)
                }
            }

            return analysis_summary
        except (ValidationError, ExternalAPIError, InternalServerError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"分析视频评论情感时出错: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"分析视频评论情感时发生未预期错误: {str(e)}")
            raise InternalServerError(detail=f"分析视频评论情感时发生未预期错误: {str(e)}")

    def analyze_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析情感分布"""
        sentiment_counts = df['sentiment'].value_counts()
        total_comments = len(df)

        return {
            'distribution': {
                sentiment: {
                    'count': int(count),
                    'percentage': round(count / total_comments * 100, 2)
                }
                for sentiment, count in sentiment_counts.items()
            },
            'dominant_sentiment': sentiment_counts.index[0],
            'sentiment_ratio': {
                'positive_ratio': round(len(df[df['sentiment'] == 'positive']) / total_comments * 100, 2),
                'negative_ratio': round(len(df[df['sentiment'] == 'negative']) / total_comments * 100, 2),
                'neutral_ratio': round(len(df[df['sentiment'] == 'neutral']) / total_comments * 100, 2)
            }
        }

    def analyze_emotion_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析情绪模式"""
        emotion_counts = df['emotion'].value_counts()
        sentiment_emotion_matrix = pd.crosstab(df['sentiment'], df['emotion'])

        return {
            'dominant_emotions': {
                'overall': emotion_counts.index[0],
                'by_sentiment': {
                    sentiment: sentiment_emotion_matrix.loc[sentiment].idxmax()
                    for sentiment in sentiment_emotion_matrix.index
                }
            },
            'emotion_distribution': {
                emotion: int(count)
                for emotion, count in emotion_counts.items()
            },
            'emotion_sentiment_correlation': {
                sentiment: {
                    emotion: int(count)
                    for emotion, count in row.items()
                }
                for sentiment, row in sentiment_emotion_matrix.iterrows()
            }
        }

    def analyze_engagement_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析互动模式"""
        engagement_counts = df['engagement_type'].value_counts()
        engagement_sentiment = pd.crosstab(df['engagement_type'], df['sentiment'])

        return {
            'engagement_distribution': {
                engagement: int(count)
                for engagement, count in engagement_counts.items()
            },
            'engagement_by_sentiment': {
                engagement: {
                    sentiment: int(count)
                    for sentiment, count in row.items()
                }
                for engagement, row in engagement_sentiment.iterrows()
            },
            'key_findings': {
                'most_common_engagement': engagement_counts.index[0],
                'least_common_engagement': engagement_counts.index[-1],
                'positive_engagement_rate': round(
                    engagement_sentiment.loc['supportive', 'positive'] /
                    engagement_sentiment.loc['supportive'].sum() * 100, 2
                )
            }
        }

    def analyze_themes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析关键主题"""
        # 展开主题列表
        all_themes = [theme for themes in df['key_themes'] for theme in themes]
        theme_counts = Counter(all_themes)

        # 按情感分类的主题
        theme_by_sentiment = {
            'positive': [
                theme for i, themes in df[df['sentiment'] == 'positive']['key_themes'].items()
                for theme in themes
            ],
            'negative': [
                theme for i, themes in df[df['sentiment'] == 'negative']['key_themes'].items()
                for theme in themes
            ],
            'neutral': [
                theme for i, themes in df[df['sentiment'] == 'neutral']['key_themes'].items()
                for theme in themes
            ]
        }

        return {
            'overall_themes': {
                theme: count
                for theme, count in theme_counts.most_common()
            },
            'themes_by_sentiment': {
                sentiment: dict(Counter(themes).most_common(5))
                for sentiment, themes in theme_by_sentiment.items()
            },
            'key_insights': {
                'top_themes': list(dict(theme_counts.most_common(5)).keys()),
                'theme_diversity': len(theme_counts),
                'average_themes_per_comment': round(len(all_themes) / len(df), 2)
            }
        }

