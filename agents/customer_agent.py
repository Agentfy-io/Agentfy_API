# -*- coding: utf-8 -*-
"""
@file: customer_agent.py
@desc: 处理TikTok评论的代理类，提供评论获取、分析和潜在客户识别功能
@auth: Callmeiks
"""

import json
import re
import sys
import time
import asyncio
from datetime import datetime
from time import process_time
from typing import Dict, Any, List, Optional, Union, AsyncGenerator

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
from services.crawler.video_crawler import VideoCollector
from app.core.exceptions import ValidationError, ExternalAPIError, InternalServerError

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
        """
        self.total_customers = 0

        # 初始化AI模型客户端
        self.chatgpt = ChatGPT()
        self.claude = Claude()

        # 初始化收集器和清洁器
        self.comment_collector = CommentCollector(tikhub_api_key)
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

            'purchase_intent_report': """You are a social commerce analyst specializing in conversion optimization. Your task is to create an actionable report analyzing comment data to reveal audience interests, product sentiment, and purchase considerations.        
                    Report Sections:
                    1. Report Metadata (video ID, total number of comments, analysis timestamp)
                    2. Audience Interest Overview (2-3 sentences summarizing what aspects of the product commenters focus on most)
                    3. Key Metrics (markdown table with statistics on product features mentioned, price sentiments, and overall attitude)
                    4. Product Aspect Analysis (analyze which product features/attributes receive the most attention and whether feedback is positive/negative)
                    5. Price Perception (analyze how commenters perceive pricing - too high, reasonable, good value, etc.)
                    6. Sales Approach Feedback (identify what selling approaches commenters respond to or suggest)
                    7. Conversion Barriers & Opportunities (specific insights about what might prevent or encourage purchases)
                    8. Recommendations (2-3 actionable steps to improve conversion based on comment analysis)
                    
                    Focus on actionable insights derived directly from the input data. Identify patterns in what commenters care about, their suggestions for selling, price perceptions, and overall product sentiment (positive or negative). Analyze variables that reflect audience opinions and highlight product effect analysis based strictly on the input data. Keep the report under 450 words.
                    """,

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
                  "messages": {
                    "jessica1h": "Customer message 1 in any language",
                    "adam_123": "Customer message 2 in any language",
                    ........
                  }
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
                    "customer_unique_id": "customer_unique_id_1",
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

    """---------------------------------------------通用方法/工具类方法---------------------------------------------"""
    async def _analyze_aspect(
            self,
            aspect_type: str,
            comment_data: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        通用分析方法，根据不同的分析类型调用ChatGPT或Claude AI模型。

        步骤:
        1. 验证分析类型是否支持
        2. 构造分析提示
        3. 调用AI模型进行分析
        4. 解析并返回分析结果

        Args:
            aspect_type: 需要分析的类型 (purchase_intent)
            comment_data: 需要分析的评论列表

        Returns:
            Optional[List[Dict[str, Any]]]: AI返回的分析结果，失败时抛出异常

        Raises:
            ValidationError: 当aspect_type无效时
            ExternalAPIError: 当调用AI服务时出错
        """
        try:
            # 验证分析类型是否支持
            if aspect_type not in self.analysis_types:
                raise ValidationError(detail=f"不支持的分析类型: {aspect_type}", field="aspect_type")

            # 检查评论数据是否为空
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

            # 尝试使用ChatGPT进行分析
            try:
                response = await self.chatgpt.chat(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt
                )

                # 解析ChatGPT返回的结果
                analysis_results = response["choices"][0]["message"]["content"].strip()

            except ExternalAPIError as e:
                # ChatGPT失败时尝试使用Claude作为备份
                logger.warning(f"ChatGPT分析失败，尝试使用Claude: {str(e)}")
                try:
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

    async def generate_analysis_report(self, aweme_id: str, analysis_type: str, data: Dict[str, Any]) -> str:
        """
        生成报告并转换为HTML

        Args:
            aweme_id (str): 视频 ID
            analysis_type (str): 分析类型
            data (Dict[str, Any]): 分析数据

        Returns:
            str: HTML报告的本地文件URL
        """
        if analysis_type not in self.system_prompts:
            raise ValueError(f"Invalid report type: {analysis_type}. Choose from {self.system_prompts.keys()}")

        # 获取系统提示
        sys_prompt = self.system_prompts[analysis_type]

        # 获取用户提示
        user_prompt = f"Generate a report for the {analysis_type} analysis based on the following data:\n{json.dumps(data, ensure_ascii=False)}, for video ID: {aweme_id}"

        # 生成报告
        response = await self.chatgpt.chat(
            system_prompt=sys_prompt,
            user_prompt=user_prompt
        )

        report = response["choices"][0]["message"]["content"].strip()

        # 保存Markdown报告
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)

        report_md_path = os.path.join(report_dir, f"report_{aweme_id}.md")
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(report)

        # 转换为HTML
        html_content = self.convert_markdown_to_html(report, f"{analysis_type.title()} Analysis for {aweme_id}")
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

        print(f"报告已生成: Markdown ({report_md_path}), HTML ({html_path})")
        print(f"报告本地URL: {file_url}")

        return file_url

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

    """---------------------------------------------获取视频评论-----------------------------------------------"""

    async def fetch_video_comments(
            self,
            aweme_id: str,
            ins_filter: bool = False,
            twitter_filter: bool = False,
            region_filter: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        获取指定视频的清理后的评论数据

        Args:
            aweme_id: 视频ID
            ins_filter: 是否过滤Instagram为None的评论
            twitter_filter: 是否过滤Twitter为None的评论
            region_filter: 评论区域过滤器，例如"US"，"GB"等

        Returns:
            Dict[str, Any]: 清理后的评论数据，包含视频ID和评论列表

        Raises:
            ValidationError: 当aweme_id为空或无效时
            ExternalAPIError: 当网络连接失败时
        """
        start_time = time.time()
        processing_time = 0
        comments = []
        total_comments = 0

        try:
            # 验证输入参数
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            # 记录开始获取评论
            logger.info(f"🔍 开始获取视频 {aweme_id} 的评论")

            # 获取评论
            async for comment_batch in self.comment_collector.stream_video_comments(aweme_id):
                # 对每批评论进行清洗
                cleaned_comments = await self.comment_cleaner.clean_video_comments(comment_batch)

                # 转换为DataFrame便于处理
                comments_df = pd.DataFrame(cleaned_comments)

                # 应用过滤条件
                if ins_filter:
                    comments_df = comments_df[comments_df['ins_id'] != '']
                if twitter_filter:
                    comments_df = comments_df[comments_df['twitter_id'] != '']
                if region_filter:
                    comments_df = comments_df[comments_df['commenter_region'] == region_filter]

                # 计算处理时间
                processing_time = round((time.time() - start_time) * 1000, 2)
                total_comments += len(comments_df)

                comments.extend(comments_df.to_dict(orient='records'))

                yield {
                    'aweme_id': aweme_id,
                    'is_complete': False,
                    'current_batch_count': len(comments_df),
                    'current_batch_comments': comments_df.to_dict(orient='records'),
                    'comments': comments,
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': processing_time
                }

            # 记录获取评论结束
            yield {
                'aweme_id': aweme_id,
                'is_complete': True,
                'total_comments': total_comments,
                'comments': comments,
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time
            }

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"获取视频评论时出错: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"获取视频评论时发生未预期错误: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'error': str(e),
                'total_comments': total_comments,
                'comments': comments,
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time
            }

    """---------------------------------------------获取购买意愿客户信息-----------------------------------------"""

    async def stream_potential_customers(
            self,
            aweme_id: str,
            customer_count: int = 100,
            min_score: float = 50.0,
            max_score: float = 100.0,
            ins_filter: bool = False,
            twitter_filter: bool = False,
            region_filter: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式获取潜在客户

        步骤:
        1. 流式获取视频评论
        2. 批量分析评论，识别潜在客户
        3. 实时产出分析结果

        Args:
            aweme_id: 视频ID
            batch_size: 每批处理的评论数量
            customer_count: 最大潜在客户数量
            concurrency: AI分析的并发数
            min_score: 最小参与度分数
            max_score: 最大参与度分数
            ins_filter: 是否过滤Instagram为None的评论
            twitter_filter: 是否过滤Twitter为None的评论
            region_filter: 评论区域过滤器

        Yields:
            Dict[str, Any]: 每批潜在客户信息

        Raises:
            ValidationError: 当aweme_id为空或无效时
            ExternalAPIError: 当网络连接失败时
        """
        start_time = time.time()
        potential_customers = []  # 临时存储分析结果

        try:
            # 验证输入参数
            if not aweme_id or not isinstance(aweme_id, str):
                raise ValidationError(detail="aweme_id必须是有效的字符串", field="aweme_id")

            logger.info(f"🔍 开始流式获取视频 {aweme_id} 的潜在客户")

            # 流式获取评论
            async for comments_batch in self.comment_collector.stream_video_comments(aweme_id):
                # 如果已经达到目标客户数量，则停止处理
                if len(potential_customers) >= customer_count:
                    logger.info(f"已达到目标客户数量 {customer_count}，停止处理")
                    break
                # 清洗评论批次
                cleaned_batch = await self.comment_cleaner.clean_video_comments(comments_batch)

                # 应用过滤条件
                if cleaned_batch:
                    batch_df = pd.DataFrame(cleaned_batch)
                    if ins_filter and 'ins_id' in batch_df.columns:
                        batch_df = batch_df[batch_df['ins_id'] != '']
                    if twitter_filter and 'twitter_id' in batch_df.columns:
                        batch_df = batch_df[batch_df['twitter_id'] != '']
                    if region_filter and 'commenter_region' in batch_df.columns:
                        batch_df = batch_df[batch_df['commenter_region'] == region_filter]
                    if batch_df.empty:
                        logger.warning(f"评论批次为空或被过滤，跳过处理")
                        continue

                # 准备分析数据
                analysis_data = [
                    {'text': comment.get('text', ''), 'comment_id': comment.get('comment_id', '')}
                    for comment in cleaned_batch
                ]
                logger.info(f"准备分析评论批次: {len(analysis_data)} 条评论")

                analysis_results = await self._analyze_aspect('purchase_intent', analysis_data)

                if analysis_results:
                    # 将分析结果与原始评论合并
                    result_df = pd.DataFrame(analysis_results)
                    batch_df = pd.DataFrame(cleaned_batch)

                    if not result_df.empty and 'comment_id' in result_df.columns:
                        # 合并分析结果
                        merged_df = pd.merge(
                            batch_df,
                            result_df,
                            on='comment_id',
                            how='inner',
                            suffixes=('', '_analysis')
                        )

                        # 过滤无效评论
                        merged_df = merged_df.drop_duplicates('commenter_uniqueId')
                        logger.info(f"合并分析结果: {len(merged_df)} 条评论")

                        # 计算参与度分数
                        merged_df['engagement_score'] = merged_df.apply(
                            lambda row: self._calculate_engagement_score(
                                row.get('sentiment', 'neutral'),
                                row.get('purchase_intent', False),
                                row.get('interest_level', 'low')
                            ),
                            axis=1
                        )

                        # 过滤符合分数范围的客户
                        filtered_df = merged_df[
                            (merged_df['engagement_score'] >= min_score) &
                            (merged_df['engagement_score'] <= max_score)
                            ]

                        # 检查是否超过客户限制并截断
                        remaining = customer_count - self.total_customers
                        if len(filtered_df) > remaining:
                            filtered_df = filtered_df.head(remaining)
                            is_complete = True
                            self.total_customers = customer_count
                            self.comment_collector.status = False
                            self.comment_cleaner.status = False
                            logger.info(f"已达到最大客户数量 {customer_count}，停止处理")
                        else:
                            is_complete = False
                            self.total_customers += len(filtered_df)

                        # 转换为列表并记录
                        filtered_list = filtered_df.to_dict('records')
                        potential_customers.extend(filtered_list)
                        logger.info(f"已处理评论 {len(merged_df)} 条，潜在客户 {len(filtered_df)} 个")

                        # 返回结果
                        yield {
                            'aweme_id': aweme_id,
                            'is_complete': is_complete,
                            'current_batch_customers': filtered_list,
                            'potential_customers': potential_customers,
                            'customer_count': len(filtered_list),
                            'timestamp': datetime.now().isoformat()
                        }

        except (ValidationError, ExternalAPIError) as e:
            # 直接向上传递这些已处理的错误
            logger.error(f"流式获取潜在客户时出错: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"流式获取潜在客户时发生未预期错误: {str(e)}")
            yield {
                'aweme_id': aweme_id,
                'error': str(e),
                'potential_customers': potential_customers,  # 返回已处理的客户
                'customer_count': len(potential_customers),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

    async def stream_keyword_potential_customers(
            self,
            keyword: str,
            customer_count: int = 100,
            min_score: float = 50.0,
            max_score: float = 100.0,
            ins_filter: bool = False,
            twitter_filter: bool = False,
            region_filter: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式获取关键词相关的潜在客户

        步骤:
        1. 获取与关键词相关的视频
        2. 并发处理多个视频的评论，识别潜在客户
        3. 实时产出分析结果

        Args:
            keyword: 关键词
            customer_count: 最大潜在客户数量
            video_concurrency: 视频处理并发数
            min_score: 最小参与度分数
            max_score: 最大参与度分数
            ins_filter: 是否过滤Instagram为空用户
            twitter_filter: 是否过滤Twitter为空用户
            region_filter: 地区过滤

        Yields:
            Dict[str, Any]: 每批潜在客户信息

        Raises:
            ValueError: 当参数无效时
            RuntimeError: 当分析过程中出现错误时
        """
        start_time = time.time()
        self.total_customers = 0
        all_potential_customers = []
        processed_videos = 0

        try:
            # 验证输入参数
            if not keyword or not isinstance(keyword, str):
                raise ValueError("无效的关键词")

            logger.info(f"🔍 开始流式获取关键词 '{keyword}' 相关视频的潜在客户")

            # 获取清理后的视频数据
            video_collector = VideoCollector(self.tikhub_api_key)
            video_cleaner = VideoCleaner()
            raw_videos = await video_collector.collect_videos_by_keyword(keyword)
            cleaned_videos = await video_cleaner.clean_videos_by_keyword(raw_videos)

            # 提取视频ID列表
            videos_df = pd.DataFrame(cleaned_videos.get('videos', []))

            if videos_df.empty:
                logger.warning(f"未找到与关键词 '{keyword}' 相关的视频")
                yield {
                    'keyword': keyword,
                    'message': f"未找到与关键词 '{keyword}' 相关的视频",
                    'potential_customers': [],
                    'customer_count': 0,
                    'timestamp': datetime.now().isoformat()
                }
                return

            aweme_ids = videos_df['aweme_id'].tolist()
            logger.info(f"找到与关键词 '{keyword}' 相关的 {len(aweme_ids)} 个视频")

            for aweme_id in aweme_ids:
                if len(all_potential_customers) >= customer_count:
                    logger.info(f"已达到目标客户数量 {customer_count}，停止处理")
                    break
                async for result in self.stream_potential_customers(
                        aweme_id,
                        customer_count=customer_count,
                        min_score=min_score,
                        max_score=max_score,
                        ins_filter=ins_filter,
                        twitter_filter=twitter_filter,
                        region_filter=region_filter
                ):
                    processed_videos += 1
                    users_list = []
                    if result.get('error'):
                        users_list = result['potential_customers']
                    else:
                        users_list = result['current_batch_customers']

                    all_potential_customers.extend(users_list)

                    yield {
                        'keyword': keyword,
                        'is_complete': False,
                        'aweme_id': result.get('aweme_id', ''),
                        'potential_customers': users_list,
                        'customer_count': len(users_list),
                        'timestamp': datetime.now().isoformat()
                    }

                    # 检查是否达到目标客户数量
                    if len(all_potential_customers) >= customer_count:
                        logger.info(f"已达到目标客户数量 {customer_count}，停止处理")
                        break
            yield {
                'keyword': keyword,
                'is_complete': True,
                'potential_customers': all_potential_customers,
                'total_customers': len(all_potential_customers),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"流式获取关键词相关潜在客户时发生未预期错误: {str(e)}")
            yield {
                'keyword': keyword,
                'error': str(e),
                'potential_customers': all_potential_customers,
                'total_customers': len(all_potential_customers),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2),
                'is_complete': True
            }


async def main():
    """测试流式关键词潜在客户功能"""
    print("开始测试 stream_keyword_potential_customers 方法...")

    # 初始化 CustomerAgent
    api_key = os.getenv("TIKHUB_API_KEY")
    if not api_key:
        print("错误: 未设置 TIKHUB_API_KEY 环境变量")
        return

    agent = CustomerAgent(api_key)

    # 测试关键词列表
    keywords = [
        "skincare products",  # 护肤产品
        #"fitness equipment"  # 健身设备
    ]

    # 对每个关键词进行测试
    for keyword in keywords:
        print(f"\n===== 测试关键词: '{keyword}' =====")

        try:
            start_time = time.time()
            batch_count = 0
            total_customers = 0

            # 流式获取关键词潜在客户
            async for result in agent.stream_keyword_potential_customers(
                    keyword=keyword,
                    customer_count=20,  # 目标客户数量
                    min_score=50.0,  # 最小参与度分数
                    max_score=100.0,  # 最大参与度分数
                    ins_filter=False,  # 不过滤Instagram
                    twitter_filter=False,  # 不过滤Twitter
                    region_filter=None  # 不过滤地区
            ):
                # 显示批次信息
                if 'is_complete' in result:
                    # 这是最终完成的结果
                    elapsed_time = time.time() - start_time
                    print(f"\n完成处理 - 总计 {result['total_customers']} 个客户")
                    print(f"处理了 {result.get('videos_processed')} 个视频 (共 {result.get('total_videos')} 个)")
                    print(f"总处理时间: {elapsed_time:.2f}秒")
                    print(f"API报告处理时间: {result.get('processing_time_ms', 0) / 1000:.2f}秒")
                elif 'error' in result:
                    # 发生错误
                    print(f"\n处理出错: {result['error']}")
                elif 'potential_customers' in result:
                    # 正常批次结果
                    batch_count += 1
                    new_customers = len(result.get('potential_customers', []))
                    total_customers = result.get('total_count', total_customers + new_customers)
                    processed_videos = result.get('total_videos_processed', 0)
                    total_videos = result.get('total_videos', 0)

                    print(f"\n批次 {batch_count}: 获得 {new_customers} 个新客户, 累计: {total_customers}")
                    print(f"正在处理视频 {processed_videos}/{total_videos}")

                    # 显示这批客户的简要信息
                    if new_customers > 0:
                        print("客户信息预览:")
                        for i, customer in enumerate(result['potential_customers'][:2]):  # 只显示前2个
                            print(f"  客户 {i + 1}:")
                            print(f"    ID: {customer.get('commenter_uniqueId', 'N/A')}")
                            print(f"    评论: {customer.get('text', 'N/A')[:40]}..." if len(
                                customer.get('text', '')) > 40 else f"    评论: {customer.get('text', 'N/A')}")
                            print(f"    参与度: {customer.get('engagement_score', 'N/A')}")
                            print(f"    来源视频: {customer.get('aweme_id', 'N/A')}")

                    # 每5个批次显示一个进度摘要
                    if batch_count % 5 == 0:
                        elapsed = time.time() - start_time
                        print(f"\n--- 进度摘要 ---")
                        print(f"已处理 {batch_count} 批次，获得 {total_customers} 个客户")
                        print(f"耗时: {elapsed:.2f}秒，平均每批次 {elapsed / batch_count:.2f}秒")
                        print(
                            f"平均每客户 {elapsed / total_customers:.2f}秒" if total_customers > 0 else "尚未获得客户")

        except Exception as e:
            print(f"测试关键词 '{keyword}' 时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n测试完成!")


if __name__ == "__main__":
    # 设置异步事件循环策略
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 运行测试
    asyncio.run(main())