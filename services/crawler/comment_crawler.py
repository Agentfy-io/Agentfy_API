import asyncio
import json
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import aiohttp
import os
from dotenv import load_dotenv

from app.config import settings
from app.utils.logger import setup_logger
from app.core.exceptions import ExternalAPIError, ValidationError, RateLimitError
from services.cleaner.video_cleaner import VideoCleaner
from services.crawler.video_crawler import VideoCollector
from services.cleaner.comment_cleaner import CommentCleaner

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
        self.status = True
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

    async def get_total_comment(self, aweme_id:str)-> Optional[int]:
        """
        获取视频的总评论数

        Args:
            aweme_id: 视频ID

        Returns:
            评论数
        """
        try:
            logger.info(f"开始获取视频 {aweme_id} 的总评论数")
            video_crawler = VideoCollector(self.api_key)
            video_cleaner = VideoCleaner()

            video_info = await video_crawler.collect_single_video(aweme_id)
            cleaned_video = await video_cleaner.clean_single_video(video_info['video'])

            count = cleaned_video['video']['statistics']['comment_count']
            return count
        except Exception as e:
            logger.error(f"获取视频 {aweme_id} 的总评论数失败: {str(e)}")
            return None

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
            'count': 30,
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

    async def collect_video_comments(self, aweme_id: str, concurrency: int = 3) -> Dict[str, Any]:
        """
        收集特定视频的评论

        Args:
            aweme_id: 视频ID
            concurrency: 并发请求数, 默认为5

        Returns:
            包含视频ID和评论列表的字典

        Raises:
            ValidationError: 当aweme_id无效时
            ExternalAPIError: 当调用外部API出错时
            RateLimitError: 当遇到速率限制时
        """
        if not aweme_id:
            raise ValidationError(detail="视频ID不能为空", field="aweme_id")

        # 获取视频总评论数
        total_comments = await self.get_total_comment(aweme_id)
        if total_comments == 0:
            # 没有评论可拉
            return {
                'aweme_id': aweme_id,
                'comments': [],
                'comment_count': 0
            }

        # 准备存储和分页控制
        comments: List[Any] = []
        chunk_size = 20              # 每次请求最多获取的评论数量
        base_cursor = 0             # 本批次并发请求的起始 cursor
        has_more = True             # 是否还有更多数据可以拉取

        try:
            async with aiohttp.ClientSession() as session:
                while self.status and has_more and len(comments) < total_comments:
                    tasks = []
                    # 一次并发 concurrency 个请求，每个请求对应不同 cursor
                    for i in range(concurrency):
                        current_cursor = base_cursor + i * chunk_size
                        tasks.append(self.fetch_comments(aweme_id, current_cursor, session))

                    # 并行执行这 concurrency 个请求
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # 这轮请求后是否还能继续
                    batch_has_more = False

                    for idx, result in enumerate(results):
                        if isinstance(result, Exception):
                            # 根据业务需求决定遇到部分请求失败时的处理策略
                            # 这里简单记录一下异常，然后跳过
                            # 也可选择 raise 终止
                            logger.error(f"cursor={base_cursor + idx * chunk_size} 请求异常：{result}")
                            continue

                        data = result['data']
                        comments_data = data['comments']
                        comments.extend(comments_data)
                        # 如果还不到总评论数，且有更多，则保持 batch_has_more = True
                        if data.get("has_more", False) and len(comments) < total_comments:
                            batch_has_more = True

                        logger.info(
                            f"已收集视频 {aweme_id} 的 {len(comments)} 条评论，"
                            f"cursor: {base_cursor + idx * chunk_size}"
                        )

                    # 如果本轮没有更多了，或者已达到/超过 total_comments，则终止循环
                    if not batch_has_more or len(comments) >= total_comments:
                        logger.info(f"完成收集视频 {aweme_id} 的评论")
                        break

                    # 更新下次并发请求的起点，让其跳到下一批次区间
                    base_cursor += concurrency * chunk_size

                    # 避免过频繁调用，可根据需求调大或调小
                    await asyncio.sleep(2)

        except (ValidationError, ExternalAPIError, RateLimitError):
            # 对已知错误直接向上抛出
            raise
        except Exception as e:
            # 捕获所有其他未预期的异常并包装成 ExternalAPIError
            logger.info(f"收集评论时发生未预期错误: {str(e)}")
            raise ExternalAPIError(
                detail="收集评论时出现未预期错误",
                service="TikHub",
                original_error=e
            )

        # 返回结构化响应
        return {
            'aweme_id': aweme_id,
            'comments': comments,
            'comment_count': len(comments)
        }



async def main():
    collector = CommentCollector()

    # 调用 collect_video_comments，并设置并发数 concurrency = 5
    # 你可以根据需要修改 concurrency 来测试不同的并发配置
    result = await collector.collect_video_comments("7469470434006715670", concurrency=5)

    print("采集结果:")
    print("视频ID:", result["aweme_id"])
    print("评论总数:", result["comment_count"])
    print("评论详情示例:", result["comments"][:5])  # 只打印前 5 条，以免输出过长


# 入口
if __name__ == "__main__":
    asyncio.run(main())