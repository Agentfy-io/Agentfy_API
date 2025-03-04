import asyncio
import json
from pathlib import Path
import aiofiles
import aiohttp
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from app.utils.logger import setup_logger
from app.config import settings

# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()

class VideoCollector:
    def __init__(self):
        self.headers = {
            'Authorization': f'Bearer {os.getenv("TIKHUB_API_KEY")}',
            'Content-Type': 'application/json'
        }
        self.base_url = os.getenv('TIKHUB_BASE_URL')
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
        """通用请求处理方法"""
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                async with session.get(url, params=params, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        wait_time = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited. Waiting {wait_time} seconds.")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await response.text()
                        logger.error(f"{error_message}: {response.status}\n{error_text}")
                        return None
            except Exception as e:
                logger.error(f"Request exception: {str(e)}")
                retries += 1
                if retries < self.MAX_RETRIES:
                    await asyncio.sleep(2 ** retries)
        return None

    async def collect_single_video(self, aweme_id:str) -> Any | None:
        """
        Collect single video data

        Args:
            aweme_id (str): The unique identifier for the video

        Returns:
            Dict: The video data
        """
        if not aweme_id:
            raise ValueError("aweme_id is required")

        async with aiohttp.ClientSession() as session:
            result = await self._make_request(
                session,
                self.endpoints['one_video'],
                {'aweme_id': aweme_id},
                f"Error fetching video {aweme_id}"
            )

            if not result:
                return None

            video_data = result.get('data', {}).get('aweme_details', [])[0]
            if not video_data:
                logger.error(f"Video {aweme_id} not found")
                return None

            return video_data

    async def collect_hashtag_videos(self, chi_id: str, batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        收集话题标签的视频，每次批量并发请求多个游标位置

        Args:
            chi_id (str): 话题标签ID
            batch_size (int): 每批并发请求的游标数量，默认5

        Returns:
            List[Dict[str, Any]]: 收集到的视频列表
        """
        if not chi_id:
            raise ValueError("chi_id is required")

        videos = []
        count_per_request = 20
        current_cursor = 0

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
                        f"Error fetching videos for hashtag {chi_id}"
                    ))

                # 并发执行所有任务
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 处理返回的结果
                any_has_more = False
                new_videos_count = 0

                for i, result in enumerate(results):
                    if isinstance(result, Exception) or not result:
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
                        logger.info(f"收集完成，游标 {cursor} 处的请求返回has_more=False")
                        return videos

                logger.info(f"收集到 {new_videos_count} 个新视频，总计 {len(videos)} 个视频")

                # 如果没有新视频，可能已经到达尽头
                if new_videos_count == 0 or not any_has_more:
                    break

                # 更新下一批次的起始游标
                current_cursor = current_cursor + (batch_size * count_per_request)

                # 批次间的速率限制
                await asyncio.sleep(1)

        return videos

    async def collect_keywords_videos(self, keyword: str, sort_type: int = 0, publish_time: int = 0,
                                      batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        收集关键词搜索的视频，每次批量并发请求多个偏移位置

        Args:
            keyword (str): 搜索关键词
            sort_type (int): 排序类型，0-相关性，1-点赞数
            publish_time (int): 发布时间，0-全部，1-天，7-周，30-月
            batch_size (int): 每批并发请求的数量，默认5

        Returns:
            List[Dict[str, Any]]: 收集到的视频列表
        """
        if not keyword:
            raise ValueError("keyword is required")

        videos = []
        count_per_request = 20
        current_offset = 0

        async with aiohttp.ClientSession() as session:
            while True:
                # 创建当前批次的任务
                tasks = []
                for i in range(batch_size):
                    offset = current_offset + (i * count_per_request)
                    tasks.append(self._make_request(
                        session,
                        self.endpoints['keywords'],
                        {
                            'keyword': keyword,
                            'offset': offset,
                            'sort_type': sort_type,
                            'publish_time': publish_time,
                            'count': count_per_request
                        },
                        f"Error fetching videos for keyword {keyword}"
                    ))

                # 并发执行所有任务
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 处理返回的结果
                new_videos_count = 0

                for i, result in enumerate(results):
                    if isinstance(result, Exception) or not result:
                        continue

                    result_data = result.get('data', {})
                    batch_videos = result_data.get('data', [])

                    if batch_videos:
                        videos.extend(batch_videos)
                        new_videos_count += len(batch_videos)

                    # 检查当前请求的has_more状态
                    if not result_data.get('has_more', False):
                        # 只要有一个请求返回has_more为False，立即停止整个收集过程
                        logger.info(f"收集完成，偏移量 {offset} 处的请求返回has_more=False")
                        return videos

                logger.info(f"收集到 {new_videos_count} 个新视频，总计 {len(videos)} 个视频")

                # 如果没有新视频，可能已经到达尽头
                if new_videos_count == 0:
                    break

                # 更新下一批次的起始偏移量
                current_offset = current_offset + (batch_size * count_per_request)

                # 批次间的速率限制
                await asyncio.sleep(1)

        return videos

