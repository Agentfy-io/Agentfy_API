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


class CustomerAgent:
    """处理TikTok评论的代理类，提供评论获取、分析和潜在客户识别功能"""

    def __init__(self, tikhub_api_key: Optional[str] = None, tikhub_base_url: Optional[str] = None):
        """
        初始化CustomerAgent，加载API密钥和提示模板

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
            'batch_customer_reply':"""## Multilingual Batch Customer Service AI Assistant
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

    async def get_purchase_intent_stats(
            self,
            aweme_id: str,
            batch_size: int = 30,
            concurrency: int = 5
    ) -> Dict[str, Any]:
        """
        获取指定视频的购买意图统计信息

        Args:
            aweme_id (str): 视频ID
            batch_size (int, optional): 每批处理的评论数量，默认30
            concurrency (int, optional): ai处理并发数，默认5

        Returns:
            Dict[str, Any]: 购买意图统计信息

        Raises:
            ValidationError: 当aweme_id为空或无效时
            ExternalAPIError: 当调用外部服务出错时
            InternalServerError: 当内部处理出错时
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

            logger.info(f"开始分析视频 {aweme_id} 的 {len(comments_df)} 条评论")
            analyzed_df = await self.analyze_comments_batch(
                comments_df,
                'purchase_intent',
                batch_size,
                concurrency
            )

            if analyzed_df.empty:
                raise InternalServerError(f"分析视频 {aweme_id} 的评论失败")

            # 生成分析摘要
            analysis_summary = {
                'sentiment_distribution': self._analyze_sentiment_distribution(analyzed_df),
                'purchase_intent': self._analyze_purchase_intent(analyzed_df),
                'interest_levels': self._analyze_interest_levels(analyzed_df),
                'meta': {
                    'total_analyzed_comments': len(analyzed_df),
                    'aweme_id': aweme_id,
                    'analysis_type': 'purchase_intent_stats',
                    'analysis_timestamp': datetime.now().isoformat(),
                    'processing_time_ms': round((time.time() - start_time) * 1000, 2)
                }
            }

            return analysis_summary

        except (ValidationError, ExternalAPIError, InternalServerError):
            # 直接向上传递这些已处理的错误
            raise
        except Exception as e:
            logger.error(f"获取购买意图统计时发生未预期错误: {str(e)}")
            raise InternalServerError(f"获取购买意图统计时发生未预期错误: {str(e)}")

    def _analyze_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析情感分布

        Args:
            df (pd.DataFrame): 包含sentiment列的DataFrame

        Returns:
            Dict[str, Any]: 情感分布统计信息
        """
        try:
            logger.info("分析情感分布")
            if 'sentiment' not in df.columns:
                raise ValueError("DataFrame必须包含'sentiment'列")

            # 标准化情感值
            df['sentiment'] = df['sentiment'].str.lower()
            df.loc[df['sentiment'] == 'neg', 'sentiment'] = 'negative'
            df.loc[df['sentiment'] == 'pos', 'sentiment'] = 'positive'

            sentiment_counts = df['sentiment'].value_counts()
            return {
                'counts': sentiment_counts.to_dict(),
                'percentages': (sentiment_counts / len(df) * 100).round(2).to_dict()
            }
        except Exception as e:
            logger.error(f"分析情感分布时出错: {str(e)}")
            return {
                'error': str(e),
                'counts': {},
                'percentages': {}
            }

    def _analyze_purchase_intent(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析购买意图

        Args:
            df (pd.DataFrame): 包含purchase_intent和interest_level列的DataFrame

        Returns:
            Dict[str, Any]: 购买意图统计信息
        """
        try:
            logger.info("分析购买意图")
            required_columns = ['purchase_intent', 'interest_level']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"DataFrame必须包含'{col}'列")

            # 确保purchase_intent是布尔值
            if df['purchase_intent'].dtype != bool:
                df['purchase_intent'] = df['purchase_intent'].apply(
                    lambda x: x if isinstance(x, bool) else (
                            str(x).lower() in ['true', 'yes', '1', 't']
                    )
                )

            # 标准化interest_level
            df['interest_level'] = df['interest_level'].str.lower()
            df.loc[df['interest_level'] == 'mid', 'interest_level'] = 'medium'

            intent_df = df[df['purchase_intent'] == True]

            return {
                'total_comments': len(df),
                'intent_count': len(intent_df),
                'intent_rate': round(len(intent_df) / len(df) * 100, 2) if len(df) > 0 else 0,
                'intent_by_interest_level': intent_df['interest_level'].value_counts().to_dict()
            }
        except Exception as e:
            logger.error(f"分析购买意图时出错: {str(e)}")
            return {
                'error': str(e),
                'total_comments': 0,
                'intent_count': 0,
                'intent_rate': 0,
                'intent_by_interest_level': {}
            }

    def _analyze_interest_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析兴趣水平

        Args:
            df (pd.DataFrame): 包含interest_level列的DataFrame

        Returns:
            Dict[str, Any]: 兴趣水平统计信息
        """
        try:
            logger.info("分析兴趣水平")
            if 'interest_level' not in df.columns:
                raise ValueError("DataFrame必须包含'interest_level'列")

            # 标准化interest_level
            df['interest_level'] = df['interest_level'].str.lower()
            df.loc[df['interest_level'] == 'mid', 'interest_level'] = 'medium'

            interest_counts = df['interest_level'].value_counts()
            return {
                'counts': interest_counts.to_dict(),
                'percentages': (interest_counts / len(df) * 100).round(2).to_dict()
            }
        except Exception as e:
            logger.error(f"分析兴趣水平时出错: {str(e)}")
            return {
                'error': str(e),
                'counts': {},
                'percentages': {}
            }

    async def get_potential_customers(
            self,
            aweme_id: str,
            batch_size: int = 30,
            concurrency: int = 5,
            min_score: float = 50.0,
            max_score: float = 100.0,
            ins_filter: bool = False,
            twitter_filter: bool = False,
            region_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        计算潜在客户的参与度分数，并识别潜在客户

        Args:
            aweme_id (str): 视频ID
            batch_size (int, optional): 每批处理的评论数量，默认30
            concurrency (int, optional): ai处理并发数，默认5
            min_score (float, optional): 最小参与度分数，默认50.0
            max_score (float, optional):f 最大参与度分数，默认100.0
            ins_filter (bool, optional): 是否过滤Instagram为Null用户，默认False
            twitter_filter (bool, optional): 是否过滤Twitter为Null用户，默认False
            region_filter (Optional[str], optional): 过滤特定地区的用户，默认None

        Returns:
            Dict[str, Any]: 潜在客户信息

        Raises:
            ValueError: 当参数无效时
            RuntimeError: 当分析过程中出现错误时
        """
        try:
            # 参数验证
            if not aweme_id:
                raise ValueError("aweme_id不能为空")

            if min_score < 0 or max_score > 100 or min_score >= max_score:
                raise ValueError("分数范围无效，应该满足: 0 <= min_score < max_score <= 100")


            # 获取清理后的评论数据
            cleaned_comments = await self.fetch_video_comments(aweme_id)

            if not cleaned_comments.get('comments'):
                logger.warning(f"视频 {aweme_id} 没有评论数据")
                return {
                    'aweme_id': aweme_id,
                    'error': 'No comments found',
                    'analysis_timestamp': datetime.now().isoformat(),
                }

            cleaned_comments = cleaned_comments.get('comments', [])
            comments_df = pd.DataFrame(cleaned_comments)

            # 过滤条件
            if ins_filter:
                comments_df = comments_df[comments_df['ins_id']!= '']
            if twitter_filter:
                comments_df = comments_df[comments_df['twitter_id']!= '']
            if region_filter:
                comments_df = comments_df[comments_df['commenter_region'] == region_filter]
            if comments_df.empty:
                logger.warning(f"视频 {aweme_id} 没有符合过滤条件的评论")
                return {
                    'aweme_id': aweme_id,
                    'error': 'No comments found after filtering',
                    'analysis_timestamp': datetime.now().isoformat(),
                }
            logger.info(f"开始分析视频 {aweme_id} 的 {len(comments_df)} 条评论以识别潜在客户")

            analyzed_df = await self.analyze_comments_batch(
                comments_df,
                'purchase_intent',
                batch_size,
                concurrency
            )

            if analyzed_df.empty:
                raise RuntimeError(f"分析视频 {aweme_id} 的评论失败")

            # 计算每个用户的参与度分数
            potential_customers = []
            for _, row in analyzed_df.iterrows():
                try:
                    potential_value = self._calculate_engagement_score(
                        row['sentiment'],
                        row['purchase_intent'],
                        row['interest_level']
                    )

                    # 检查必填字段
                    user_id = row.get('commenter_uniqueId', '')
                    sec_uid = row.get('commenter_secuid', '')
                    text = row.get('text')

                    #if not all([user_id, sec_uid, text]):
                    #    logger.warning(f"评论数据缺少必要字段，跳过: {row}")
                    #    continue

                    potential_customers.append({
                        'user_uniqueId': user_id,
                        'potential_value': potential_value,
                        'user_secuid': sec_uid,
                        'ins_id': row.get('ins_id', ''),
                        'twitter_id': row.get('twitter_id', ''),
                        'region': row.get('commenter_region', ''),
                        'text': text
                    })
                except Exception as e:
                    logger.error(f"处理用户评论时出错: {str(e)}, 跳过该评论")
                    continue

            # 创建潜在客户DataFrame
            if not potential_customers:
                logger.warning(f"未找到任何潜在客户")
                return {
                    'aweme_id': aweme_id,
                    'total_potential_customers': 0,
                    'min_engagement_score': min_score,
                    'max_engagement_score': max_score,
                    'average_potential_value': 0,
                    'potential_customers': []
                }

            potential_customers_df = pd.DataFrame(potential_customers)

            # 根据潜在价值过滤和排序
            filtered_df = potential_customers_df[
                potential_customers_df['potential_value'].between(min_score, max_score)
            ].sort_values(by='potential_value', ascending=False)

            avg_value = filtered_df['potential_value'].mean() if not filtered_df.empty else 0

            return {
                'aweme_id': aweme_id,
                'total_potential_customers': len(filtered_df),
                'min_engagement_score': min_score,
                'max_engagement_score': max_score,
                'average_potential_value': round(avg_value, 2),
                'potential_customers': filtered_df.to_dict(orient='records'),
                'analysis_timestamp': datetime.now().isoformat()
            }

        except ValueError:
            # 直接向上传递验证错误
            raise ValueError
        except RuntimeError:
            # 直接向上传递运行时错误
            raise RuntimeError
        except Exception as e:
            logger.error(f"获取潜在客户时发生未预期错误: {str(e)}")
            raise RuntimeError(f"获取潜在客户时发生未预期错误: {str(e)}")


    def _calculate_engagement_score(
            self,
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

    async def get_keyword_potential_customers(
            self,
            keyword: str,
            batch_size: int = 30,
            video_concurrency: int = 5,
            ai_concurrency: int = 5,
            min_score: float = 50.0,
            max_score: float = 100.0,
            ins_filter: bool = False,
            twitter_filter: bool = False,
            region_filter: Optional[str] = None,
            max_customers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        根据关键词获取潜在客户

        Args:
            keyword (str): 关键词
            batch_size (int, optional): 每批处理的评论数量，默认30
            video_concurrency (int, optional): 视频处理并发数，默认5
            ai_concurrency (int, optional): ai处理并发数，默认5
            min_score (float, optional): 最小参与度分数，默认50.0
            max_score (float, optional): 最大参与度分数，默认100.0
            ins_filter (bool, optional): 是否过滤Instagram为空用户，默认False
            twitter_filter (bool, optional): 是否过滤Twitter为空用户，默认False
            region_filter (str, optional): 地区过滤，默认None
            max_customers (int, optional): 最大潜在客户数量，默认None, None表示不限制

        Returns:
            Dict[str, Any]: 潜在客户信息

        Raises:
            ValueError: 当参数无效时
            RuntimeError: 当分析过程中出现错误时
        """
        try:
            # 参数验证
            if not keyword:
                raise ValueError("keyword不能为空")

            if min_score < 0 or max_score > 100 or min_score >= max_score:
                raise ValueError("分数范围无效，应该满足: 0 <= min_score < max_score <= 100")

            # 获取清理后的视频数据
            video_collector = VideoCollector(self.tikhub_api_key, self.tikhub_base_url)
            video_cleaner = VideoCleaner()
            raw_videos = await video_collector.collect_videos_by_keyword(keyword)
            cleaned_videos = await video_cleaner.clean_videos_by_keyword(raw_videos)

            if not cleaned_videos.get('videos'):
                logger.warning(f"未找到与关键词 {keyword} 相关的视频")
                return {
                    'keyword': keyword,
                    'error': '没有找到相关视频',
                    'analysis_timestamp': datetime.now().isoformat(),
                }

            videos_df = pd.DataFrame(cleaned_videos['videos'])
            aweme_ids = videos_df['aweme_id'].tolist()

            logger.info(f"开始分析与关键词 {keyword} 相关的 {len(videos_df)} 个视频以识别潜在客户")

            # 使用get_potential_customers方法分析每个视频，每concurrency个视频为一组，
            potential_customers = []
            for i in range(0, len(aweme_ids), video_concurrency):
                batch_aweme_ids = aweme_ids[i:i + video_concurrency]
                tasks = [
                    self.get_potential_customers(
                        aweme_id,
                        batch_size,
                        ai_concurrency,
                        min_score,
                        max_score,
                        ins_filter,
                        twitter_filter,
                        region_filter,
                    ) for aweme_id in batch_aweme_ids
                ]

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"处理视频 {batch_aweme_ids} 时出错: {str(result)}")
                        continue

                    if result.get('potential_customers') is None:
                        logger.warning(f"视频 {batch_aweme_ids} 没有潜在客户数据")
                        continue

                    potential_customers.extend(result['potential_customers'])
                    if len(potential_customers) >= max_customers:
                        break
                if len(potential_customers) >= max_customers:
                    break
            # 创建潜在客户DataFrame
            if not potential_customers:
                logger.warning(f"未找到任何潜在客户")
                return {
                    'keyword': keyword,
                    'total_potential_customers': 0,
                    'min_engagement_score': min_score,
                    'max_engagement_score': max_score,
                    'average_potential_value': 0,
                    'potential_customers': []
                }

            potential_customers_df = pd.DataFrame(potential_customers)
            # 根据潜在价值过滤和排序
            filtered_df = potential_customers_df[
                potential_customers_df['potential_value'].between(min_score, max_score)
            ].sort_values(by='potential_value', ascending=False)

            avg_value = filtered_df['potential_value'].mean() if not filtered_df.empty else 0

            return {
                'keyword': keyword,
                'total_potential_customers': len(filtered_df),
                'min_engagement_score': min_score,
                'max_engagement_score': max_score,
                'average_potential_value': round(avg_value, 2),
                'potential_customers': filtered_df.to_dict(orient='records'),
                'analysis_timestamp': datetime.now().isoformat()
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

    async def generate_single_reply_message(
            self,
            shop_info: str,
            customer_id: str,
            customer_message: str,
    ) -> Dict[str, Any]:
        """
        生成单条客户回复消息

        Args:
            shop_info (str): 店铺信息
            customer_id (str): 客户uniqueID
            customer_message (str): 客户消息
        Returns:
            Dict[str, Any]: 生成的回复消息

        Raises:
            ValueError: 当参数无效时
            RuntimeError: 当分析过程中出现错误时
        """
        try:
            # 参数验证
            if not customer_message:
                raise ValueError("customer_message不能为空")

            sys_prompt = self.system_prompts['customer_reply']
            user_prompt = f"Here is the shop information:\n{shop_info}\n\nHere is the customer message:\n{customer_message},\n\nPlease generate a reply message for the customer."

            # 生成回复消息
            reply_message = await self.chatgpt.chat(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
            )

            # 解析回复消息
            reply_message = reply_message["choices"][0]["message"]["content"].strip()
            # 解析json
            reply_message = re.sub(
                r"```json\n|\n```",
                "",
                reply_message.strip()
            )  # 去除Markdown代码块

            reply_message = json.loads(reply_message)

            return {
                'customer_id': customer_id,
                'reply_message': reply_message,
            }
        except ValueError:
            # 直接向上传递验证错误
            raise ValueError
        except RuntimeError:
            # 直接向上传递运行时错误
            raise RuntimeError
        except Exception as e:
            logger.error(f"生成单条客户回复消息时发生未预期错误: {str(e)}")
            raise RuntimeError(f"生成单条客户回复消息时发生未预期错误: {str(e)}")

    async def generate_customer_reply_messages(
            self,
            shop_info: str,
            customer_messages: List[Dict[str, Any]],
            batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        批量生成客户回复消息

        Args:
            shop_info (str): 店铺信息
            customer_messages (List[Dict[str, Any]]): 客户消息列表, 每个消息包括commenter_uniqueId, comment_id, text
            batch_size (int, optional): 每批处理的客户消息数量. 默认为5.

        Returns:
            List[Dict[str, Any]]: 生成的回复消息列表, 每个回复包含语言检测和回复内容

        Raises:
            ValueError: 当参数无效时
            RuntimeError: 当分析过程中出现错误时
        """
        try:
            # 参数验证
            if not shop_info:
                raise ValueError("店铺信息不能为空")

            if not customer_messages:
                raise ValueError("客户消息列表不能为空")

            # 检查消息格式
            for msg in customer_messages:
                if "commenter_uniqueId" not in msg or "text" not in msg:
                    raise ValueError(f"消息格式错误, 必须包含commenter_uniqueId和text字段: {msg}")

            # 准备结果列表
            all_replies = []

            # 按批次处理消息
            for i in range(0, len(customer_messages), batch_size):
                batch = customer_messages[i:i + batch_size]

                # 构建批处理提示
                batch_messages = []
                for idx, msg in enumerate(batch):
                    batch_messages.append({
                        "message_id": idx,
                        "commenter_uniqueId": msg.get("commenter_uniqueId"),
                        "comment_id": msg.get("comment_id", ""),
                        "message_text": msg.get("text")  # 注意此处key改为message_text以适配prompt
                    })

                batch_prompt = {
                    "shop_info": shop_info,
                    "messages": batch_messages
                }

                # 将字典转换为JSON字符串
                batch_prompt_json = json.dumps(batch_prompt, ensure_ascii=False)

                # 调用AI生成回复
                batch_replies = await self.chatgpt.chat(
                    system_prompt=self.system_prompts['batch_customer_reply'],
                    user_prompt=batch_prompt_json,  # 确保这里传入的是字符串
                    temperature=0.7
                )

                # 解析AI回复
                batch_replies = batch_replies["choices"][0]["message"]["content"].strip()
                # 解析JSON
                batch_replies = re.sub(
                    r"```json\n|\n```",
                    "",
                    batch_replies.strip()
                )

                # 解析回复结果
                try:
                    parsed_replies = json.loads(batch_replies)

                    # 验证回复格式
                    if not isinstance(parsed_replies, list):
                        raise ValueError("AI返回的结果格式错误，应为列表")

                    # 将批次回复添加到总结果中
                    for reply in parsed_replies:
                        # 确保uniqueID在回复中
                        message_id = reply.get("message_id")
                        if message_id is not None and 0 <= message_id < len(batch):
                            reply["commenter_uniqueId"] = batch[message_id].get("commenter_uniqueId")
                            reply["comment_id"] = batch[message_id].get("comment_id", "")

                        all_replies.append(reply)

                except json.JSONDecodeError as json_err:
                    logger.error(f"无法解析AI返回的JSON结果: {batch_replies[:200]}... (错误: {str(json_err)})")
                    raise RuntimeError(f"AI返回的结果不是有效的JSON格式: {str(json_err)}")

            return all_replies

        except ValueError as e:
            logger.warning(f"参数验证错误: {str(e)}")
            raise

        except RuntimeError as e:
            logger.error(f"运行时错误: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"批量生成客户回复消息时发生未预期错误: {str(e)}", exc_info=True)
            raise RuntimeError(f"批量生成客户回复消息时发生未预期错误: {str(e)}")

async def main():
    # 创建CustomerAgent实例
    agent = CustomerAgent()

    # 示例：获取keyword潜在客户
    #keyword = "red liptick"
    #potential_customers = await agent.get_keyword_potential_customers(keyword,20, 5, 5, 0, 100.0, True, False, 'US', 100)

    #save to json
    #with open('potential_customers.json', 'w', encoding='utf-8') as f:
    #    json.dump(potential_customers, f, ensure_ascii=False, indent=4)

    # 生成单条客户回复消息测试
    shop_info = """
    店铺名称：优雅时尚屋
店铺简介：优雅时尚屋成立于2015年，专注于提供高品质的时尚服装和配饰。我们致力于为客户提供最新的时尚趋势和永恒的经典款式。

产品信息：
- 女士连衣裙：价格范围在￥299-￥899，材质包括棉、丝绸和混纺面料
- 男士衬衫：价格范围在￥199-￥599，有多种款式和颜色可选
- 时尚包包：价格范围在￥499-￥1899，有真皮和高级PU材质可选
- 精美首饰：价格范围在￥99-￥699，包括项链、耳环和手链

促销活动：
- 新用户首单满￥500减￥50
- 每周三会员日，全场9折
- 季末清仓，指定商品低至5折

配送政策：
- 国内订单满￥199免运费，否则运费￥15
- 国际配送可发往亚洲、欧洲、北美等地区，运费根据重量和目的地计算
- 正常订单处理时间为1-2个工作日，国内配送3-5天，国际配送7-15天

退换政策：
- 收到商品后7天内可申请退换
- 商品需保持原包装和吊牌完好
- 定制商品和特价商品不支持退换

支付方式：
- 支持支付宝、微信支付、银联卡和主流国际信用卡
- 国际订单支持PayPal支付

联系方式：
- 客服电话：400-888-7777（工作日9:00-18:00）
- 客服邮箱：service@elegantfashion.com
- 微信公众号：优雅时尚屋"""
    #customer_id = "12345"
    #customer_message = "Où puis-je trouver les informations sur la livraison ?"
    #reply_lang = "zh"

    #reply_message = await agent.generate_single_reply_message(shop_info, customer_id, customer_message)
    #print(reply_message)

    # 生成批量客户回复消息测试
    customer_messages = [
        {
            "commenter_uniqueId": "user1",
            "comment_id": "c1",
            "text": "Où puis-je trouver les informations sur la livraison ?"  # 法语: 我在哪里可以找到配送信息？
        },
        {
            "commenter_uniqueId": "user2",
            "comment_id": "c2",
            "text": "What is the return policy?"  # 英语: 退货政策是什么？
        },
        {
            "commenter_uniqueId": "user3",
            "comment_id": "c3",
            "text": "¿Cuánto tiempo tarda en llegar mi pedido a España?"  # 西班牙语: 我的订单多久能送到西班牙？
        },
        {
            "commenter_uniqueId": "user4",
            "comment_id": "c4",
            "text": "この商品は日本に配送できますか？送料はいくらですか？"  # 日语: 这个商品可以送到日本吗？运费是多少？
        },
        {
            "commenter_uniqueId": "user5",
            "comment_id": "c5",
            "text": "Ich möchte wissen, ob die Größe M noch verfügbar ist?"  # 德语: 我想知道M码是否还有库存？
        },
        {
            "commenter_uniqueId": "user6",
            "comment_id": "c6",
            "text": "Ho fatto un ordine tre giorni fa ma non ho ricevuto nessuna conferma. Potete aiutarmi?"
            # 意大利语: 我三天前下了订单但没收到确认。您能帮我吗？
        },
        {
            "commenter_uniqueId": "user7",
            "comment_id": "c7",
            "text": "Принимаете ли вы оплату PayPal?"  # 俄语: 你们接受PayPal付款吗？
        },
        {
            "commenter_uniqueId": "user8",
            "comment_id": "c8",
            "text": "我想了解一下这个产品的材质是什么？是纯棉的吗？"  # 中文: 询问产品材质
        },
        {
            "commenter_uniqueId": "user9",
            "comment_id": "c9",
            "text": "Do you offer express shipping? I need this item by next week."  # 英语: 你们提供快递吗？我需要下周收到这个物品。
        },
        {
            "commenter_uniqueId": "user10",
            "comment_id": "c10",
            "text": "هل تشحنون إلى المملكة العربية السعودية؟ وكم تكلفة الشحن؟"  # 阿拉伯语: 你们发货到沙特阿拉伯吗？运费是多少？
        }
    ]

    batch_reply_messages = await agent.generate_customer_reply_messages(shop_info, customer_messages, 10)
    print(batch_reply_messages)



if __name__ == "__main__":
    asyncio.run(main())


