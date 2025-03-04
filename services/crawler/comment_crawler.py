import asyncio
import json
from typing import Dict, List, Any, Optional, Union
import aiohttp
import os
from dotenv import load_dotenv

from app.config import settings
from app.utils.logger import setup_logger
from app.core.exceptions import ExternalAPIError, ValidationError, RateLimitError

# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()


class CommentCollector:
    """TikTok评论收集器，负责从TikHub API获取视频评论"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化评论收集器

        Args:
            api_key: TikHub API密钥，如果不提供则使用环境变量中的默认值
            base_url: TikHub API基础URL，如果不提供则使用环境变量中的默认值
        """
        self.api_key = api_key or settings.TIKHUB_API_KEY
        self.base_url = base_url or settings.TIKHUB_BASE_URL

        if not self.api_key:
            logger.warning("未提供TikHub API密钥，某些功能可能不可用")

        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.base_url = f"{self.base_url.rstrip('/')}/api/v1/tiktok/app/v2/fetch_video_comments"
        self.MAX_RETRIES = 3

    async def fetch_comments(
            self,
            aweme_id: str,
            cursor: int,
            session: aiohttp.ClientSession
    ) -> Optional[Dict[str, Any]]:
        """
        获取特定视频的评论

        Args:
            aweme_id: 视频ID
            cursor: 分页游标
            session: aiohttp会话

        Returns:
            评论数据字典

        Raises:
            RateLimitError: 当遇到速率限制时
            ExternalAPIError: 当调用外部API出错时
        """
        params = {
            'aweme_id': aweme_id,
            'count': 20,
            'cursor': cursor
        }

        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                async with session.get(self.base_url, params=params, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 401:
                        # 无效的API密钥
                        error_text = await response.text()
                        logger.error(f"TikHub API密钥无效: {error_text}")
                        raise ExternalAPIError(
                            detail="TikHub API密钥无效或已过期",
                            service="TikHub",
                            status_code=401
                        )
                    elif response.status == 429:  # 速率限制
                        wait_time = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"遇到速率限制。等待 {wait_time} 秒后重试。")

                        # 对于API调用，直接抛出错误，让上层处理速率限制
                        if retries == self.MAX_RETRIES - 1:
                            raise RateLimitError(
                                detail=f"TikHub API速率限制。请等待 {wait_time} 秒后重试。",
                                retry_after=wait_time
                            )

                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await response.text()
                        logger.error(f"获取视频 {aweme_id} 评论时出错: HTTP {response.status}, {error_text}")

                        if response.status >= 500:
                            # 服务器错误，重试
                            retries += 1
                            if retries < self.MAX_RETRIES:
                                await asyncio.sleep(2 ** retries)  # 指数退避
                                continue

                        # 对于其他错误，直接抛出
                        raise ExternalAPIError(
                            detail=f"调用TikHub API出错: HTTP {response.status}",
                            service="TikHub",
                            status_code=response.status
                        )

            except aiohttp.ClientError as e:
                logger.error(f"获取评论时发生网络错误: {str(e)}")
                retries += 1
                if retries < self.MAX_RETRIES:
                    await asyncio.sleep(2 ** retries)  # 指数退避
                else:
                    raise ExternalAPIError(
                        detail="连接到TikHub API时出错",
                        service="TikHub",
                        original_error=e
                    )

            except (RateLimitError, ExternalAPIError):
                # 直接将这些错误向上传递
                raise

            except Exception as e:
                logger.error(f"获取评论时发生未预期错误: {str(e)}")
                retries += 1
                if retries < self.MAX_RETRIES:
                    await asyncio.sleep(2 ** retries)
                else:
                    raise ExternalAPIError(
                        detail="获取TikTok评论时出现未预期错误",
                        service="TikHub",
                        original_error=e
                    )

        # 如果所有重试都失败
        raise ExternalAPIError(
            detail=f"获取视频 {aweme_id} 评论失败，已达到最大重试次数",
            service="TikHub"
        )

    async def collect_video_comments(self, aweme_id: str) -> Dict[str, Any]:
        """
        收集特定视频的评论

        Args:
            aweme_id: 视频ID

        Returns:
            包含视频ID和评论列表的字典

        Raises:
            ValidationError: 当aweme_id无效时
            ExternalAPIError: 当调用外部API出错时
            RateLimitError: 当遇到速率限制时
        """
        if not aweme_id:
            raise ValidationError(detail="视频ID不能为空", field="aweme_id")

        cursor = 0
        comments = []
        has_more = True

        try:
            async with aiohttp.ClientSession() as session:
                while has_more:
                    data = await self.fetch_comments(aweme_id, cursor, session)

                    if not data:
                        logger.error(f"获取视频 {aweme_id} 评论时出错: 无响应数据")
                        break

                    data = data.get('data', {})
                    comments_data = data.get('comments', [])
                    comments.extend(comments_data)
                    logger.info(f"已收集视频 {aweme_id} 的 {len(comments)} 条评论，游标: {cursor}")

                    cursor = data.get('cursor', 0)
                    has_more = data.get('has_more', False)

                    if not has_more:
                        logger.info(f"完成收集视频 {aweme_id} 的评论")
                        break

                    # 速率限制
                    await asyncio.sleep(1)

        except (ValidationError, ExternalAPIError, RateLimitError):
            # 直接向上传递这些已知错误
            raise

        except Exception as e:
            logger.error(f"收集评论时发生未预期错误: {str(e)}")
            raise ExternalAPIError(
                detail="收集TikTok评论时出现未预期错误",
                service="TikHub",
                original_error=e
            )

        # 返回结构化响应
        return {
            'aweme_id': aweme_id,
            'comments': comments,
            'comment_count': len(comments)
        }