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
from services.ai_models.genny import Genny
from app.config import settings
from app.core.exceptions import ValidationError, ExternalAPIError, InternalServerError
from services.auth.auth_service import user_sessions

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
        self.genny = Genny()

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
            "text_to_script": """You are a professional short-video script creation assistant. When given a brief request from the user (e.g., "Create a short story about Nezha"), you should automatically infer and decide on the following five elements:           
            1. language:
               - The language to be used in the script. 
               - Must be returned using an ISO language-region format (e.g., "zh-CN", "en-US"). 
               - If the user does not specify a language, you should automatically match the user's input or default to an appropriate language.           
            2. context:
               - The general context or theme (Mythology & Folklore, Historical Events & Biographies, Science & Technology, Educational/Academic, Art & Literature, Culture & Heritage, Travel & Exploration, Food & Culinary, Product Promotion/Commercial, Lifestyle, Entertainment & Pop Culture, Inspirational & Motivational, Health & Wellness, Personal Development & Self-Help, Philosophy & Ethics, Finance & Economics, Business & Entrepreneurship, Marketing & Advertising, Environmental & Sustainability, Social Issues & Awareness, Comedy & Satire, Parenting & Family, Relationships & Dating, Case Studies/Testimonials, Professional Skills & Career, Language Learning, Hobbies & Crafts, Gaming, News & Current Events, Sports & Fitness, Music & Performing Arts, Film & TV)                  
            3. scenarioType:
               - The scenario type. (Storytelling,Product Showcase, Educational/Explainer,Tutorial/How-To, Promotional/Marketing, Testimonial/Review, Comedy/Entertainment, Lifestyle/Vlog, Inspirational/Motivational, Documentary-Style, News/Current Events, Case Study, Interview or Q&A, Health & Wellness, Cooking/Recipe,Fashion/Beauty, Travel/Adventure, Tech Tips/Hacks, Challenge or Game.)              
            4. tone:
               - The overall style or mood (friendly, serious, playful, comedic, dramatic, authoritative, casual, formal, lively, motivational, urgent, enthusiastic, calm, warm, uplifting, confident, nostalgic, encouraging, soothing, empathetic, matter-of-fact, neutral, romantic, humorous). 
               - If the user does not specify, default to a neutral/friendly tone.           
            5. duration:
               - The target reading duration for the script. If the user does not specify a duration, please aim for approximately 30 seconds of spoken content.          
            **Your task**:
            - When the user provides only a brief request, infer as many details as possible for these eight elements. 
            - Create a short script suitable for platforms like TikTok or any short-video format, including an engaging opening, core content, and a concise conclusion or call-to-action. 
            - Use textual cues to indicate pacing or tempo if needed (e.g., ellipses for dramatic pauses).
            - Ensure the script is engaging, informative, and suitable for the specified context and language.
            - you may pair up multiple tones with different scenarios, but make sure the tone is appropriate for the context.
            - Return **only** the following JSON structure (with no extra text or formatting):            

            {
                "text": "Place the final short-video (or audio) script here",
                "metadata": {
                    "language": "ISO language-region code (e.g. zh-CN, en-US)",
                    "context": "e.g. mythological story",
                    "scenarioType": "e.g. storytelling",
                    "tone": "e.g. narrative and friendly",
                    "duration": "e.g. 30 seconds"
                }
            }                        
            **Important**:
            - Make sure the returned field names and structure match exactly.
            - If any user requirement is unclear or missing, assume reasonable defaults. 
            - Always ensure the final script matches the language and context inferred or specified by the user.           
            - Remember that this transcript will be converted to speech, so it should sound natural when read aloud. Use appropriate pacing, transitions, and conversational elements to ensure a smooth listening experience in the specified language.
            """
        }

    def _load_user_prompts(self) -> None:
        """加载用户提示用于不同语音生成类型"""
        pass


    async def text_to_script(self, prompt: str, scenarioType: str, language:str ) -> dict[str, Any]:
        """
        根据用户输入关键词生成语音文本

        Args:
            prompt: 用户提示
            scenarioType: 场景类型
            language: 语言

        Returns:
            转换后的文本
        """
        start_time = time.time()

        try:
            if not prompt or not scenarioType or not language:
                raise ValidationError("缺少必要参数")
            logger.info(f"开始生成语音文本")
            # 生成文本
            sys_prompt = self.system_prompts.get("text_to_script")
            user_prompt = f"Here is the request from the user: {prompt}\n\n, the scenario type is {scenarioType}, the language is {language}"
            response = await self.chatgpt.chat(
                system_prompt=sys_prompt,
                user_prompt=user_prompt
            )

            # 解析ChatGPT返回的结果
            transcript = response["choices"][0]["message"]["content"].strip()

            logger.info(f"生成语音文本成功，耗时: {time.time() - start_time:.2f}秒")
            return {
                "transcript": transcript,
                "metadata": {
                    "language": "zh-CN",
                    "context": "storytelling",
                    "speed": "medium",
                    'generated_at': datetime.now().isoformat(),
                    'processing_time': time.time() - start_time
                }
            }
        except ValidationError:
            # 直接向上传递验证错误
            raise
        except ExternalAPIError:
            # 直接向上传递API错误
            raise
        except Exception as e:
            logger.error(f"分析评论方面时发生未预期错误: {str(e)}")
            raise InternalServerError(f"分析评论方面时发生未预期错误: {str(e)}")

    async def script_to_audio(self, text: str, language: str, gender: str, age: str, speed: int = 1) -> Dict[str, Any]:
        """
        根据自定义text生成Tiktok音频

        Args:
            text: 待转换的文本
            language: 语言
            gender: 性别
            age: 年龄
            speed: 语速

        Returns:
            生成的音频文件信息
        """

        start_time = time.time()
        try:
            logger.info(f"开始生成音频")
            # 获取发音人列表
            speakers = await self.genny.get_speakers(gender, age, language)

            if len(speakers) == 0:
                logger.error("该语言和性别没有可用的发音人")
                raise ExternalAPIError("该语言和性别没有可用的发音人")

            # 选择第一个发音人
            speaker_id = speakers[0]['id']
            speaker_name = speakers[0]['displayName']

            # 生成音频文件
            audio_info = await self.genny.generate_voice(text, speaker_id, speed)

            audio_summary = {
                "audio_url": audio_info['data'][0]['urls'][0],
                "progress": audio_info['progress'],
                "status": audio_info['status'],
                "meta": {
                    "speaker": speaker_name,
                    "language": language,
                    "gender": gender,
                    "age": age,
                    "speed": speed,
                    "generated_at": datetime.now().isoformat(),
                    "processing_time": time.time() - start_time
                }
            }

            logger.info(f"生成Tiktok音频成功，耗时: {time.time() - start_time:.2f}秒")

            return audio_summary
        except ExternalAPIError as e:
            logger.error(f"无法生成Tiktok音频: {str(e)}")
            raise ExternalAPIError("无法生成Tiktok音频")
        except Exception as e:
            logger.error(f"未知错误: {str(e)}")
            raise InternalServerError("未知错误")

    async def text_to_audio(self, prompt: str, scenarioType: str, language: str, gender: str, age: str, speed: int = 1) -> Dict[str, Any]:
        """
        根据用户输入关键词生成音频

        Args:
            prompt: 用户提示
            scenarioType: 场景类型
            language: 语言
            gender: 性别
            age: 年龄
            speed: 语速

        Returns:
            转换后的文本
        """
        start_time = time.time()

        try:
            # 生成文本
            script = await self.text_to_script(prompt, scenarioType, language)

            # 生成音频
            audio = await self.script_to_audio(script['transcript'], language,gender, age, speed)

            return audio
        except ValidationError:
            # 直接向上传递验证错误
            raise
        except ExternalAPIError:
            # 直接向上传递API错误
            raise
        except Exception as e:
            logger.error(f"生成音频时发生未预期错误: {str(e)}")
            raise InternalServerError(f"生成音频时发生未预期错误: {str(e)}")


async def run_test():
    agent = AudioGeneratorAgent()
    prompt = "Create a short story about Nezha"
    scenarioType = "storytelling"
    language = "zh-CN"
    gender = "female"
    age = "young_adult"

    try:
        result = await agent.text_to_audio(prompt, scenarioType, language, gender, age)
        print(result)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    asyncio.run(run_test())

