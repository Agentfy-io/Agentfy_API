# -*- coding: utf-8 -*-
"""
@file: audio_generator.py
@desc: 为创作者生成音频文件的代理类，提供文本到语音转换功能，支持多种语言和声音效果
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
from services.ai_models.whisper import WhisperLemonFox
from app.config import settings
from app.core.exceptions import ValidationError, ExternalAPIError, InternalServerError

# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()


class AudioGeneratorAgent:
    """
    为创作者生成音频文件的代理类，提供文本到语音转换功能，支持多种语言和声音效果
    """

    def __init__(self, tikhub_api_key: Optional[str] = None, tikhub_base_url: Optional[str] = None):
        """
        初始化音频生成代理类

        Args:
            tikhub_api_key: TikHub API密钥
            tikhub_base_url: TikHub API基础URL
        """
        # 初始化AI模型客户端
        self.chatgpt = ChatGPT()
        self.claude = Claude()
        self.whisper = WhisperLemonFox()

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


    async def generate_tk_audio(self, text: str, speaker: str, speed: float = 1.0) -> Dict[str, Any]:
        """
        生成TikHub音频文件

        Args:
            text: 待转换的文本
            speaker: 说话者ID
            speed: 语速

        Returns:
            生成的音频文件信息
        """
        # 构建请求参数
        payload = {
            "speed": speed,
            "text": text,
            "speaker": speaker
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-API-KEY": self.tikhub_api_key
        }

        # 发送请求
        try:
            response = await self.whisper.tts_sync(payload, headers)
            return response
        except ExternalAPIError as e:
            logger.error(f"Failed to generate TikHub audio: {str(e)}")
            raise InternalServerError("Failed to generate TikHub audio")


