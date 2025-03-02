import json
import traceback
from typing import Dict, Any, Optional

from openai import AsyncOpenAI, OpenAIError
from app.config import settings
from app.utils.logger import setup_logger
from app.core.exceptions import ExternalAPIError

# 设置日志记录器
logger = setup_logger(__name__)


class ChatGPT:
    """OpenAI API客户端封装类，支持异步调用ChatGPT模型"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        初始化ChatGPT客户端

        Args:
            openai_api_key: OpenAI API密钥，如果未提供则从环境变量中读取
        """
        # 设置OpenAI API Key
        self.openai_key = openai_api_key or settings.OPENAI_API_KEY

        if not self.openai_key:
            logger.warning("未提供OpenAI API密钥，ChatGPT功能将不可用")
            self.openai_client = None
        else:
            # 初始化OpenAI
            self.openai_client = AsyncOpenAI(api_key=self.openai_key, timeout=60)

    async def chat(self,
                   system_prompt: str,
                   user_prompt: str,
                   model: str = None,
                   temperature: float = None,
                   max_tokens: int = None,
                   timeout: int = 60,
                   ) -> Dict[str, Any]:
        """
        调用OpenAI的聊天接口（异步）

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            model: 模型名称，默认使用配置中的DEFAULT_AI_MODEL
            temperature: 温度参数，默认使用配置中的DEFAULT_TEMPERATURE
            max_tokens: 最大生成长度，默认使用配置中的DEFAULT_MAX_TOKENS
            timeout: 超时时间，默认为60秒

        Returns:
            返回生成的结果（dict）

        Raises:
            ExternalAPIError: 当调用OpenAI API出错时
        """
        # 检查客户端是否初始化
        if not self.openai_client:
            raise ExternalAPIError(
                detail="OpenAI客户端未初始化，请提供有效的API密钥",
                service="OpenAI"
            )

        # 使用配置默认值
        model = model or settings.DEFAULT_AI_MODEL
        temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
        max_tokens = max_tokens or settings.DEFAULT_MAX_TOKENS

        try:
            # 调用OpenAI的聊天接口
            chat_completion = await self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            # 记录基本响应信息，但不包含完整内容（可能很大）
            logger.info(
                f"OpenAI响应: 模型={model}, "
                f"完成度={chat_completion.usage.completion_tokens}/{chat_completion.usage.total_tokens}"
            )

            # 返回生成的结果
            return chat_completion.model_dump()

        except OpenAIError as e:
            # 记录并封装OpenAI特定错误
            logger.error(f"OpenAI API错误: {str(e)}")
            raise ExternalAPIError(
                detail=f"调用OpenAI API时出错: {str(e)}",
                service="OpenAI",
                original_error=e
            )

        except Exception as e:
            # 记录其他未预期错误
            logger.error(f"调用OpenAI时发生未预期错误: {str(e)}")
            traceback.print_exc()
            raise ExternalAPIError(
                detail=f"调用OpenAI时发生未预期错误",
                service="OpenAI",
                original_error=e
            )