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

    async def calculate_openai_cost(self, model: str, prompt_tokens: int, completion_tokens: int)->Dict[str, Any]:
        """
        异步计算OpenAI API使用成本

        参数:
        model (str): 模型名称（如'gpt-4o', 'o1', 'gpt-4o-mini'等）
        prompt_tokens (int): 输入token数量
        completion_tokens (int): 输出token数量
        use_cached_input (bool): 是否使用缓存输入价格（默认False）

        返回:
        dict: 包含input_cost, output_cost和total_cost的字典
        """
        # 模型价格配置（每个token的美元价格）
        pricing = {
            "o1": {
                "input": 15.00 / 1000000,
                "cached_input": 7.50 / 1000000,
                "output": 60.00 / 1000000
            },
            "o3-mini": {
                "input": 1.10 / 1000000,
                "cached_input": 0.55 / 1000000,
                "output": 4.40 / 1000000
            },
            "gpt-4.5": {
                "input": 75.00 / 1000000,
                "cached_input": 37.50 / 1000000,
                "output": 150.00 / 1000000
            },
            "gpt-4o": {
                "input": 2.50 / 1000000,
                "cached_input": 1.25 / 1000000,
                "output": 10.00 / 1000000
            },
            "gpt-4o-mini": {
                "input": 0.150 / 1000000,
                "cached_input": 0.075 / 1000000,
                "output": 0.600 / 1000000
            },
            "gpt-3.5-turbo": {
                "input": 0.0015 / 1000,
                "cached_input": 0.0015 / 1000,
                "output": 0.002 / 1000
            }
        }

        # 标准化模型名称 - 这个步骤可以在异步环境中运行
        model_key = model.lower()
        if "o1" in model_key:
            model_key = "o1"
        elif "o3-mini" in model_key:
            model_key = "o3-mini"
        elif "gpt-4.5" in model_key:
            model_key = "gpt-4.5"
        elif "gpt-4o-mini" in model_key:
            model_key = "gpt-4o-mini"
        elif "gpt-4o" in model_key:
            model_key = "gpt-4o"
        elif "gpt-3.5" in model_key:
            model_key = "gpt-3.5-turbo"

        if model_key not in pricing:
            raise ValueError(f"未知模型: {model}")

        # 计算成本
        input_cost = prompt_tokens * pricing[model_key]["input"]
        output_cost = completion_tokens * pricing[model_key]["output"]
        total_cost = input_cost + output_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }

    async def chat(self,
                   system_prompt: str,
                   user_prompt: str,
                   model: str = None,
                   temperature: float = None,
                   max_tokens: int = 10000,
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

            cost = await self.calculate_openai_cost(model, chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)

            result = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response": chat_completion.model_dump(),
                "cost": cost
            }

            # 记录基本响应信息，但不包含完整内容（可能很大）
            logger.info(
                f"OpenAI响应: 模型={model}, "
                f"完成度={chat_completion.usage.completion_tokens}/{chat_completion.usage.total_tokens} "
                f"输入成本={cost['input_cost']:.6f}, 输出成本={cost['output_cost']:.6f}, 总成本={cost['total_cost']:.6f}"
            )

            # 返回生成的结果
            return result
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