# -*- coding: utf-8 -*-
"""
@file: customer_agent.py
@desc: 处理TikTok评论的代理类，提供评论获取、分析和潜在客户识别功能
@auth: Callmeiks
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import asyncio
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

from app.config import settings
from app.utils.logger import setup_logger
from services.ai_models.chatgpt import ChatGPT
from services.ai_models.claude import Claude
from services.cleaner.comment_cleaner import CommentCleaner
from services.cleaner.video_cleaner import VideoCleaner
from services.crawler.comment_crawler import CommentCollector
from app.core.exceptions import ValidationError, ExternalAPIError, InternalServerError
from services.crawler.video_crawler import VideoCollector

# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()


class CustomerAgent:
    """处理TikTok评论的代理类，提供评论获取、分析和潜在客户识别功能"""

    def __init__(self, tikhub_api_key: Optional[str] = None):
        """
        初始化CustomerAgent，加载API密钥和提示模板

        Args:
            tikhub_api_key: TikHub API密钥
            tikhub_base_url: TikHub API基础URL
        """

        self.customer_count = 0
        # 初始化AI模型客户端
        self.chatgpt = ChatGPT()
        self.claude = Claude()

        self.comment_collector = CommentCollector(tikhub_api_key, settings.TIKHUB_BASE_URL)
        self.comment_cleaner = CommentCleaner()

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
        self._load_user_prompts()

    def _load_system_prompts(self) -> None:
        """加载系统提示用于不同的评论分析类型"""
        self.system_prompts = {
            'purchase_intent': """You are an AI trained to analyze social media comments about products.
            For each comment in the provided list, analyze:
            1. Sentiment (positive, negative, or neutral)
            2. Purchase Intent (whether the user shows any interest in buying, trying, or acquiring the product)
            3. User Interest Level (high, medium, low)

            For each comment, provide analysis in the following JSON format:
            {
                "comment_id": "comment ID from input data",
                "text": "comment text",
                "sentiment": "positive/negative/neutral",
                "purchase_intent": true/false,
                "interest_level": "high/medium/low",
            }

            Respond with a JSON array containing analysis for all comments.
            Focus on identifying any signs of interest in the product, including:
            - Direct purchase intentions ("want to buy", "need this")
            - Indirect interest ("where can I find this", "love this")
            - Questions about product details
            - emojis indicating positive sentiment or interest
            - Positive reactions to product features
            - Any positive sentiment to the influencer herself/himself is consider neutral
            - Any engagement indicating potential future purchase""",
            'customer_reply': """# # Multilingual Customer Service AI Assistant

            ## System Instruction

            You are an advanced multilingual customer service AI for an e-commerce platform. Your task is to:
            1. Analyze the store information provided by the merchant
            2. Identify the language of both the store information and customer message
            3. Generate a helpful, accurate response to the customer inquiry
            4. Return your response in a structured JSON format

            ## Analysis Guidelines

            1. **Language Detection**:
               - Automatically detect the language of the store information
               - Automatically detect the language of the customer message
               - Translate the customer message to the store language
               - Determine the most appropriate language for your response (typically matching the customer's language)

            2. **Store Information Analysis**:
               - Extract key details about products, pricing, shipping, returns, promotions, etc.
               - Understand store policies across different languages
               - Identify store name and branding elements

            3. **Customer Message Analysis**:
               - Identify the customer's primary question or concern
               - Detect any secondary questions
               - Understand the customer's tone and respond appropriately

            ## Response Generation

            1. **Content Creation**:
               - Provide accurate information based only on the store details provided
               - If information is not available, acknowledge this limitation politely
               - Structure your response with greeting, answer, additional information.
               - Be concise and to the point, avoiding unnecessary details, usually one or two sentences is enough.
               - Maintain a helpful, professional tone appropriate to the detected culture
               - Also generate a translated version of your response in the shop language

            2. **Response Language**:
               - Respond in the same language as the customer message
               - If you cannot confidently respond in the customer's language, default to the store's language
               - Use culturally appropriate greetings and expressions

            ## JSON Output Format

            Your response must be formatted as a valid JSON object with the following structure:
            ```json
            {
              "detected_store_language": "language_code",
              "detected_customer_language": "language_code",
              "response_language": "language_code",
              'translated_customer_message': 'translated message',
              "response_text": "Your complete customer service response",
              "response_text_translated": "Your response translated to the store language",
              "confidence_score": 0.95
            }
            """,
            'batch_customer_reply': """## Multilingual Batch Customer Service AI Assistant
### System Instruction
You are an advanced multilingual customer service AI for an e-commerce platform. Your task is to process multiple customer messages simultaneously and generate appropriate responses for each.
### Input Format
You will receive a JSON object with the following structure:
```json
{
  "shop_info": "Complete store information in any language",
  "messages": [
    {
      "message_id": 0,
      "commenter_uniqueId": "customer_unique_id_1",
      "comment_id": "optional_comment_id_1",
      "message_text": "Customer message 1 in any language"
    },
    {
      "message_id": 1,
      "commenter_uniqueId": "customer_unique_id_2",
      "comment_id": "optional_comment_id_2",
      "message_text": "Customer message 2 in any language"
    },
    ...
  ]
}
```
### Processing Steps
For each customer message, perform the following:

#### 1. Language Detection
- Detect the language of the store information.
- Detect the language of the customer message.
- Choose the appropriate language for your response (typically the customer's language).

#### 2. Message Translation
- If the customer's language differs from the store language, translate the customer message TO THE STORE'S LANGUAGE and save it as translated_customer_message.
- This translation helps the store owner understand what the customer is saying in the store owner's language.

#### 3. Content Analysis
- Understand the store policies, products, and services based on the `shop_info`.
- Identify the customer's question or concern.
- Formulate a helpful, accurate response based only on the available information.

#### 4. Response Creation
Generate a complete, professional response in the customer's language. Include:
- **Greeting:** Friendly opening appropriate to the language/culture.
- **Answer:** Directly address the customer's question.

If necessary, translate your response into the store's language for reference.

### Output Format
Return a JSON array containing one object for each input message in the same order as received. Each object must have the following structure:

```json
[
  {
    "message_id": 0,
    "detected_store_language": "language_code",
    "detected_customer_language": "language_code",
    'customer_unique_id': 'customer_unique_id_1',
    "translated_customer_message": "translated message",
    "response_text": "Your complete customer service response in customer's language",
    "response_text_translated": "Your response translated to the store language"
  },
  {
    "message_id": 1,
    "detected_store_language": "language_code",
    "detected_customer_language": "language_code",
    'customer_unique_id': 'customer_unique_id_2',
    "translated_customer_message": "translated message",
    "response_text": "Your complete customer service response in customer's language",
    "response_text_translated": "Your response translated to the store language"
  }
]
```

### Important Rules
✅ Return **ONLY** a valid JSON array with objects matching the format above.
✅ Use **ISO 639-1** codes for language identification (e.g., "en", "zh", "fr", "es").
✅ Base responses **ONLY** on the provided store information.
✅ If the provided information is insufficient to answer a question, clearly state this.
✅ Maintain a professional and helpful tone.

### Response Content Guidelines
Each response should include:
- **Greeting** - Friendly and culturally appropriate.
- **Answer** - Direct and accurate, usually one or two sentences is enough.

This format ensures efficient multilingual customer support while maintaining high-quality, contextually relevant responses. 🚀
"""
        }

    def _load_user_prompts(self) -> None:
        """加载用户提示用于不同的评论分析类型"""
        self.user_prompts = {
            'purchase_intent': {
                'description': 'purchase intent'
            }
        }

    async def fetch_video_comments(self, aweme_id: str, ins_filter: bool = False, twitter_filter: bool = False, region_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        获取指定视频的清理后的评论数据

        Args:
            aweme_id (str): 视频ID
            ins_filter (bool): 是否过滤Instagram为None的评论
            twitter_filter (bool): 是否过滤Twitter为None的评论
            region_filter (str): 评论区域过滤器，例如"US"，"GB"等

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
            logger.info(f"🔍 开始获取视频 {aweme_id} 的评论")

            # 获取评论
            comments = await self.comment_collector.collect_video_comments(aweme_id)

            if not comments or not comments.get('comments'):
                logger.warning(f"❌ 视频 {aweme_id} 未找到评论")
                return {
                    'aweme_id': aweme_id,
                    'comments': [],
                    'comment_count': 0,
                    'timestamp': datetime.now().isoformat()
                }

            # 清洗评论
            cleaned_comments = await self.comment_cleaner.clean_video_comments(comments)
            cleaned_comments = cleaned_comments.get('comments', [])

            comments_df = pd.DataFrame(cleaned_comments)

            # 过滤条件
            if ins_filter:
                comments_df = comments_df[comments_df['ins_id']!= '']
            if twitter_filter:
                comments_df = comments_df[comments_df['twitter_id']!= '']
            if region_filter:
                comments_df = comments_df[comments_df['commenter_region'] == region_filter]

            processing_time = time.time() - start_time

            result = {
                'aweme_id': aweme_id,
                'comments': comments_df.to_dict(orient='records'),
                'comment_count': len(comments_df),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round(processing_time * 1000, 2)
            }

            logger.info(f"成功获取视频 {aweme_id} 的评论: {len(comments_df)} 条，耗时: {processing_time:.2f}秒")
            return result

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"获取视频评论时出错: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"获取视频评论时发生未预期错误: {str(e)}")
            raise InternalServerError(detail=f"获取视频评论时发生未预期错误: {str(e)}")

    async def get_potential_customers(
            self,
            aweme_id: str,
            batch_size: int = 30,
            max_count: int = 100,
            concurrency: int = 5,
            min_score: float = 50.0,
            max_score: float = 100.0,
            ins_filter: bool = False,
            twitter_filter: bool = False,
            region_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取潜在客户列表

        1. 调用fetch_video_comments方法获取视频评论
        2. 使用ins_filter，twitter_filter, region_filter 去过滤数据
        3. 使用AI模型分析符合条件评论，识别潜在客户
        4. 返回潜在客户列表
        """


        start_time = time.time()
        potential_customers = []  # 潜在客户列表

        try:
            # 获取评论数据，并且过滤+清洗
            comments_data = await self.fetch_video_comments(aweme_id, ins_filter, twitter_filter, region_filter)
            comments = comments_data.get('comments')

            # 将comments列表转换为DataFrame
            comments_df = pd.DataFrame(comments)

            # 将清洗后的评论数据按照AI并发量（concurrency）分批处理
            num_splits = max(1, len(comments_df) // batch_size + (1 if len(comments_df) % batch_size > 0 else 0))
            comment_batches = np.array_split(comments_df, num_splits)
            logger.info(
                f"🚀 开始分析评论，共 {len(comment_batches)} 批，每批约 {len(comment_batches[0]) if len(comment_batches) > 0 else 0} 条评论"
            )

            # 按照批次处理评论
            for i in range(0, len(comment_batches), concurrency):
                batch_group = comment_batches[i:i + concurrency]
                batch_indices = [
                    f"{batch.index[0]}-{batch.index[-1]}" if not batch.empty else "-"
                    for batch in batch_group
                ]
                logger.info(
                    f"⚡ 处理批次 {i + 1} 至 {i + len(batch_group)}，评论索引范围: {batch_indices}"
                )

                # 开始使用AI模型分析评论
                tasks = [
                    self._analyze_aspect('purchase_intent', batch[['text', 'comment_id']].to_dict('records'))
                    for batch in batch_group if not batch.empty
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # 处理每个批次的结果
                for j, (batch, result) in enumerate(zip(batch_group, batch_results)):
                    if isinstance(result, Exception) or not result:  # 如果结果为空或者是异常
                        logger.error(f"批次 {i + j + 1} 分析失败: {str(result)}")
                    else:
                        result_df = pd.DataFrame(result)
                        if not result_df.empty and 'comment_id' in result_df.columns:  # 如果结果不为空, 并且包含comment_id
                            # 合并分析结果到原始评论
                            merged_batch = pd.merge(
                                batch,
                                result_df,
                                on='comment_id',
                                how='left',
                                suffixes=('', '_analysis')
                            )

                            # 计算参与度分数
                            merged_batch['engagement_score'] = merged_batch.apply(
                                lambda row: self._calculate_engagement_score(
                                    row['sentiment'],
                                    row['purchase_intent'],
                                    row['interest_level']
                                ),
                                axis=1
                            )

                            # 过滤分数在min_score和max_score之间的评论
                            filtered_batch = merged_batch[
                                (merged_batch['engagement_score'] >= min_score) &
                                (merged_batch['engagement_score'] <= max_score)
                                ]

                            # 记录潜在客户数量
                            self.customer_count += len(filtered_batch)
                            logger.info(f"批次 {i + j + 1} 处理完成: 原始评论 {len(batch)}，"
                                        f"分析后 {len(merged_batch)}，过滤后 {len(filtered_batch)}")

                            # 如果客户总数超过最大限制，则只添加到最大限制
                            if self.customer_count > max_count:
                                potential_customers.extend(
                                    filtered_batch.head(max_count - self.customer_count).to_dict('records'))
                                self.comment_collector.status = False  # 停止收集评论
                                self.comment_cleaner.status = False  # 停止清洗评论
                                logger.info(
                                    f"现在已经有 {self.customer_count} 个潜在客户，达到最大限制 {max_count}，停止处理")
                                break

                            # 添加到潜在客户列表
                            potential_customers.extend(filtered_batch.to_dict('records'))
                        else:
                            logger.warning(f"批次 {i + j + 1} 分析结果格式无效或为空")
                            # 保留原始批次
                            potential_customers = batch.to_dict('records')

                # 如果客户总数超过最大限制，则停止处理
                if self.customer_count > max_count:
                    self.comment_collector.status = False  # 停止收集评论
                    self.comment_cleaner.status = False  # 停止清洗评论
                    break

            processing_time = time.time() - start_time
            logger.info(
                f"✅ 分析完成，成功处理 {len(comments)} 条评论，客户总数: {len(potential_customers)}，耗时: {processing_time:.2f}秒")
            return {
                'aweme_id': aweme_id,
                'potential_customers': potential_customers,
                'customer_count': len(potential_customers),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round(processing_time * 1000, 2)
            }
        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"获取潜在客户时出错: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"获取潜在客户时发生未预期错误: {str(e)}")
            raise InternalServerError(detail=f"获取潜在客户时发生未预期错误: {str(e)}")

    async def get_keyword_potential_customers(
            self,
            keyword: str,
            batch_size: int = 30,
            customer_count: int = 100,
            video_concurrency: int = 5,
            ai_concurrency: int = 5,
            min_score: float = 50.0,
            max_score: float = 100.0,
            ins_filter: bool = False,
            twitter_filter: bool = False,
            region_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        根据关键词获取潜在客户

        Args:
            keyword (str): 关键词
            batch_size (int, optional): 每批处理的评论数量，默认30
            customer_count (int, optional): 最大潜在客户数量，默认100
            video_concurrency (int, optional): 视频处理并发数，默认5
            ai_concurrency (int, optional): ai处理并发数，默认5
            min_score (float, optional): 最小参与度分数，默认50.0
            max_score (float, optional): 最大参与度分数，默认100.0
            ins_filter (bool, optional): 是否过滤Instagram为空用户，默认False
            twitter_filter (bool, optional): 是否过滤Twitter为空用户，默认False
            region_filter (str, optional): 地区过滤，默认None

        Returns:
            Dict[str, Any]: 潜在客户信息

        Raises:
            ValueError: 当参数无效时
            RuntimeError: 当分析过程中出现错误时
        """
        start_time = time.time()
        potential_customers = []  # 潜在客户列表

        try:
            if not keyword or not isinstance(keyword, str):
                raise ValueError("无效的关键词")

            # 获取清理后的视频数据
            video_collector = VideoCollector(self.tikhub_api_key, self.tikhub_base_url)
            video_cleaner = VideoCleaner()
            raw_videos = await video_collector.collect_videos_by_keyword(keyword)
            cleaned_videos = await video_cleaner.clean_videos_by_keyword(raw_videos)

            videos_df = pd.DataFrame(cleaned_videos['videos'])
            aweme_ids = videos_df['aweme_id'].tolist()

            if not aweme_ids:
                logger.warning(f"未找到与关键词 {keyword} 相关的视频")
                return {
                    'keyword': keyword,
                    'potential_customers': [],
                    'customer_count': 0,
                    'timestamp': datetime.now().isoformat()
                }

            logger.info(f"开始分析与关键词 {keyword} 相关的 {len(videos_df)} 个视频以识别潜在客户")

            # 按照视频并发数处理视频
            for i in range(0, len(aweme_ids), video_concurrency):
                if self.customer_count >= customer_count: # 如果已经达到最大潜在客户数量，停止处理
                    self.comment_collector.status = False  # 停止收集评论
                    self.comment_cleaner.status = False  # 停止清洗评论
                    logger.info(f"已经达到最大潜在客户数量 {customer_count}，停止处理")
                    break
                batch_aweme_ids = aweme_ids[i:i + video_concurrency]
                logger.info(f"处理视频批次 {i + 1} 至 {i + len(batch_aweme_ids)}")
                tasks = [
                    self.get_potential_customers(
                        aweme_id,
                        batch_size,
                        customer_count,
                        ai_concurrency,
                        min_score,
                        max_score,
                        ins_filter,
                        twitter_filter,
                        region_filter
                    )
                    for aweme_id in batch_aweme_ids
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception) or not result:
                        logger.error(f"视频 {aweme_ids[i + j]} 处理失败/停止处理: {str(result)}")
                    else:
                        potential_customers.extend(result.get('potential_customers', []))
                        logger.info(f"视频 {aweme_ids[i + j]} 处理完成")


            # 根据潜在价值过滤和排序
            potential_customers_df = pd.DataFrame(potential_customers)
            filtered_df = potential_customers_df[
                potential_customers_df['potential_value'].between(min_score, max_score)
            ].sort_values(by='potential_value', ascending=False)

            # 计算平均潜在价值
            avg_value = filtered_df['potential_value'].mean() if not filtered_df.empty else 0

            processing_time = time.time() - start_time
            return {
                'keyword': keyword,
                'customer_count': len(potential_customers),
                'potential_customers': potential_customers,
                'min_engagement_score': min_score,
                'max_engagement_score': max_score,
                'average_potential_value': round(avg_value, 2),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round(processing_time * 1000, 2)
            }
        except ValueError:
            # 直接向上传递验证错误
            raise ValueError
        except RuntimeError:
            # 直接向上传递运行时错误
            raise RuntimeError
        except Exception as e:
            logger.error(f"获取关键词潜在客户时发生未预期错误: {str(e)}")
            raise RuntimeError(f"获取关键词潜在客户时发生未预期错误: {str(e)}")


    async def _analyze_aspect(
            self,
            aspect_type: str,
            comment_data: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        通用分析方法，根据不同的分析类型调用ChatGPT或Claude AI模型。

        Args:
            aspect_type (str): 需要分析的类型 (purchase_intent)
            max_count (int): 最大分析数量
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

    @staticmethod
    def _calculate_engagement_score(
            sentiment: str,
            purchase_intent: bool,
            interest_level: str
    ) -> float:
        """
        计算潜在客户的参与度分数

        Args:
            sentiment (str): 情感分析结果 ('positive', 'neutral', 'negative')
            purchase_intent (bool): 是否有购买意图
            interest_level (str): 兴趣水平 ('high', 'medium', 'low')

        Returns:
            float: 参与度分数 (0-100)
        """
        try:
            # 参数验证和标准化
            if not isinstance(sentiment, str):
                sentiment = str(sentiment).lower()
            else:
                sentiment = sentiment.lower()

            if not isinstance(purchase_intent, bool):
                # 尝试转换为布尔值
                if isinstance(purchase_intent, str):
                    purchase_intent = purchase_intent.lower() in ['true', '1', 'yes', 't']
                else:
                    purchase_intent = bool(purchase_intent)

            if not isinstance(interest_level, str):
                interest_level = str(interest_level).lower()
            else:
                interest_level = interest_level.lower()

            # 修正情感标签
            if sentiment in ['neg', 'negative']:
                sentiment = 'negative'
            elif sentiment in ['pos', 'positive']:
                sentiment = 'positive'
            elif sentiment not in ['neutral']:
                sentiment = 'neutral'  # 默认为中性

            # 处理兴趣水平中值的不同表示
            if interest_level in ['mid', 'medium']:
                interest_level = 'medium'
            elif interest_level not in ['high', 'low']:
                interest_level = 'low'  # 默认为低兴趣

            # 1. 情感转换 (0-1 scale)
            sentiment_score = {
                'positive': 1.0,
                'neutral': 0.5,
                'negative': 0.0
            }.get(sentiment, 0.5)  # 默认为中性

            # 2. 购买意图分数
            intent_score = 1.0 if purchase_intent else 0.0

            # 3. 兴趣水平分数
            interest_score = {
                'high': 1.0,
                'medium': 0.5,
                'low': 0.2
            }.get(interest_level, 0.5)  # 默认为中等

            # 根据加权平均计算潜在价值
            potential_value = (
                                      0.3 * sentiment_score +
                                      0.4 * intent_score +
                                      0.3 * interest_score
                              ) * 100  # 缩放到0-100

            return round(potential_value, 2)

        except Exception as e:
            logger.error(f"计算参与度分数时出错: {str(e)}")
            # 返回默认值
            return 0.0

async def main():
    agent = CustomerAgent()
    # 获取潜在客户
    result = await agent.get_keyword_potential_customers("iphone 13", customer_count=400, min_score=50, max_score=100)

if __name__ == "__main__":
    asyncio.run(main())

