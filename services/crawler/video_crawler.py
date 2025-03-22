import asyncio
import json
from pathlib import Path
import aiofiles
import aiohttp
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import os
from dotenv import load_dotenv
from app.utils.logger import setup_logger
from app.config import settings
from app.core.exceptions import ExternalAPIError, ValidationError, RateLimitError
from services.cleaner.video_cleaner import VideoCleaner

# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()


class VideoCollector:
    """TikTok视频收集器，负责从TikHub API获取视频数据"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化视频收集器

        Args:
            api_key: TikHub API密钥，如果不提供则使用环境变量中的默认值
            base_url: TikHub API基础URL，如果不提供则使用环境变量中的默认值
        """
        self.status = True
        self.api_key = api_key
        self.base_url = settings.TIKHUB_BASE_URL

        if not self.api_key:
            logger.warning("未提供TikHub API密钥，某些功能可能不可用")

        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        self.endpoints = {
            'one_video': f"{self.base_url}/api/v1/tiktok/app/v3/fetch_one_video",
            'hashtag': f"{self.base_url}/api/v1/tiktok/app/v3/fetch_hashtag_video_list",
            'keywords': f"{self.base_url}/api/v1/tiktok/app/v3/fetch_video_search_result"
        }

        self.MAX_RETRIES = 3

    async def _make_request(
            self,
            session: aiohttp.ClientSession,
            url: str,
            params: Dict[str, Any],
            error_message: str
    ) -> Optional[Dict]:
        """
        通用请求处理方法

        Args:
            session: aiohttp会话
            url: 请求URL
            params: 请求参数
            error_message: 错误信息前缀

        Returns:
            请求响应数据

        Raises:
            RateLimitError: 当遇到速率限制时
            ExternalAPIError: 当调用外部API出错时
        """
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                async with session.get(url, params=params, headers=self.headers) as response:
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
                        logger.error(f"{error_message}: HTTP {response.status}, {error_text}")

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
                logger.error(f"请求时发生网络错误: {str(e)}")
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
                logger.error(f"请求时发生未预期错误: {str(e)}")
                retries += 1
                if retries < self.MAX_RETRIES:
                    await asyncio.sleep(2 ** retries)
                else:
                    raise ExternalAPIError(
                        detail="请求TikHub API时出现未预期错误",
                        service="TikHub",
                        original_error=e
                    )

        # 如果所有重试都失败
        raise ExternalAPIError(
            detail=f"{error_message}，已达到最大重试次数",
            service="TikHub"
        )

    async def collect_single_video(self, aweme_id: str) -> Optional[Dict[str, Any]]:
        """
        收集单个视频的数据

        Args:
            aweme_id: 视频ID

        Returns:
            视频数据字典

        Raises:
            ValidationError: 当aweme_id无效时
            ExternalAPIError: 当调用外部API出错时
            RateLimitError: 当遇到速率限制时
        """
        if not aweme_id:
            raise ValidationError(detail="视频ID不能为空", field="aweme_id")

        try:
            async with aiohttp.ClientSession() as session:
                result = await self._make_request(
                    session,
                    self.endpoints['one_video'],
                    {'aweme_id': aweme_id},
                    f"获取视频 {aweme_id} 时出错"
                )

                if not result:
                    logger.error(f"获取视频 {aweme_id} 时出错: 无响应数据")
                    return None

                video_data = result.get('data', {}).get('aweme_details', [])[0]
                if not video_data:
                    logger.error(f"未找到视频 {aweme_id}")
                    return None

                return {
                    'aweme_id': aweme_id,
                    'video': video_data,
                }

        except (ValidationError, ExternalAPIError, RateLimitError):
            # 直接向上传递这些已知错误
            raise

        except Exception as e:
            logger.error(f"获取视频时发生未预期错误: {str(e)}")
            raise ExternalAPIError(
                detail="获取TikTok视频时出现未预期错误",
                service="TikHub",
                original_error=e
            )

    async def collect_videos_by_hashtag(self, chi_id: str, batch_size: int = 5) -> Dict[str, Any]:
        """
        收集话题标签的视频，每次批量并发请求多个游标位置

        Args:
            chi_id: 话题标签ID
            batch_size: 每批并发请求的游标数量，默认5

        Returns:
            收集到的视频列表

        Raises:
            ValidationError: 当chi_id无效时
            ExternalAPIError: 当调用外部API出错时
            RateLimitError: 当遇到速率限制时
        """
        if not chi_id:
            raise ValidationError(detail="话题标签ID不能为空", field="chi_id")

        videos = []
        count_per_request = 20
        current_cursor = 0

        try:
            async with aiohttp.ClientSession() as session:
                while True:
                    # 创建当前批次的任务
                    tasks = []
                    for i in range(batch_size):
                        cursor = current_cursor + (i * count_per_request)
                        tasks.append(self._make_request(
                            session,
                            self.endpoints['hashtag'],
                            {
                                'ch_id': chi_id,
                                'cursor': cursor,
                                'count': count_per_request
                            },
                            f"获取话题 {chi_id} 的视频时出错"
                        ))

                    # 并发执行所有任务
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # 处理返回的结果
                    any_has_more = False
                    new_videos_count = 0

                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            if isinstance(result, (ExternalAPIError, RateLimitError)):
                                raise result
                            logger.error(f"批次请求发生错误: {str(result)}")
                            continue

                        if not result:
                            continue

                        result_data = result.get('data', {})
                        batch_videos = result_data.get('aweme_list', [])

                        if batch_videos:
                            videos.extend(batch_videos)
                            new_videos_count += len(batch_videos)

                        # 检查当前请求的has_more状态
                        if result_data.get('has_more', False):
                            any_has_more = True
                        else:
                            # 只要有一个请求返回has_more为False，立即停止整个收集过程
                            cursor = current_cursor + (i * count_per_request)
                            logger.info(f"收集完成，游标 {cursor} 处的请求返回has_more=False")
                            return videos

                    logger.info(f"已收集 {new_videos_count} 个新视频，总计 {len(videos)} 个视频")

                    # 如果没有新视频，可能已经到达尽头
                    if new_videos_count == 0 or not any_has_more:
                        break

                    # 更新下一批次的起始游标
                    current_cursor = current_cursor + (batch_size * count_per_request)

                    # 批次间的速率限制
                    await asyncio.sleep(1)

            return {
                'chi_id': chi_id,
                'videos': videos,
                'video_count': len(videos)
            }

        except (ValidationError, ExternalAPIError, RateLimitError):
            # 直接向上传递这些已知错误
            raise

        except Exception as e:
            logger.error(f"收集话题视频时发生未预期错误: {str(e)}")
            raise ExternalAPIError(
                detail="收集TikTok话题视频时出现未预期错误",
                service="TikHub",
                original_error=e
            )

    async def stream_videos_by_keyword(self, keyword: str, count: int = 20, concurrency: int = 5) -> AsyncGenerator[
        List[Dict], None]:
        """
        流式收集关键词搜索的视频，以批次方式产出视频

        Args:
            keyword: 搜索关键词
            count: 每次请求的视频数量，默认20
            batch_size: 每次产出的批次大小，默认10

        Yields:
            视频的批次列表

        Raises:
            ValidationError: 当keyword无效时
            ExternalAPIError: 当调用外部API出错时
            RateLimitError: 当遇到速率限制时
        """
        if not keyword:
            raise ValidationError(detail="搜索关键词不能为空", field="keyword")

        if count <= 0 or count > 30:
            raise ValidationError(detail="视频数量必须在1到30之间", field="count")

        if concurrency <= 0 or concurrency > 10:
            raise ValidationError(detail="并发请求数必须在1到10之间", field="concurrency")

        current_batch = []
        current_offset = 0
        has_more = True
        total_collected = 0

        try:
            async with aiohttp.ClientSession() as session:
                while self.status and has_more:
                    task = []
                    for i in range(concurrency):
                        task.append(self._make_request(
                            session,
                            self.endpoints['keywords'],
                            {
                                'keyword': keyword,
                                'offset': current_offset,
                                'count': count,
                                'sort_type': 0,
                                'publish_time': 0,
                            },
                            f"获取关键词 {keyword} 的视频时出错"
                        ))

                    results = await asyncio.gather(*task, return_exceptions=True)

                    batch_videos = []

                    for idx, result in enumerate(results):
                        if isinstance(result, Exception):
                            if isinstance(result, (ExternalAPIError, RateLimitError)):
                                raise result
                            logger.error(f"批次请求发生错误: {str(result)}")
                            continue

                        data = result['data']
                        awemes = data['data']
                        has_more = data['has_more']

                        if awemes:
                            batch_videos.extend(awemes)
                            total_collected += len(awemes)

                    if batch_videos:
                        logger.info(
                            f"流式产出关键词 {keyword} 的一批视频: {len(batch_videos)} 条, "
                            f"总计已收集: {total_collected}"
                        )

                        yield batch_videos

                    if has_more == False:
                        logger.info(f"收集完成，偏移量 {current_offset} 处的请求返回has_more=False")
                        break

                    current_offset += count * concurrency

        except (ValidationError, ExternalAPIError, RateLimitError):
            # 直接向上传递这些已知错误
            raise
        except Exception as e:
            logger.error(f"流式收集关键词视频时发生未预期错误: {str(e)}")
            raise ExternalAPIError(
                detail="流式收集TikTok关键词视频时出现未预期错误",
                service="TikHub",
                original_error=e
            )

async def main():
    # 创建视频收集器
    collector = VideoCollector(api_key=os.getenv("TIKHUB_API_KEY"))
    cleaner = VideoCleaner()

    # 流式收集关键词视频
    async for batch in collector.stream_videos_by_keyword("tiktok", count=10, concurrency=2):
        cleaned_video = await cleaner.clean_videos_by_keyword(batch)
        print(f"Received batch of {len(cleaned_video)} videos")



if __name__ == "__main__":
    asyncio.run(main())

