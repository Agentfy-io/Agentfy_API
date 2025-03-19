import asyncio
import json
import aiohttp
from typing import List, Dict, Any, Optional, AsyncGenerator
import os
from aiohttp import ClientSession
from dotenv import load_dotenv

from app.core.exceptions import ValidationError, ExternalAPIError, RateLimitError
from app.utils.logger import setup_logger
from app.config import settings
from services.cleaner.user_cleaner import UserCleaner

# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()


class UserCollector:
    """
    TikTok用户数据收集器
    功能：收集TikTok用户的档案信息、粉丝列表、发布的视频等数据
    主要特点：
    1. 异步请求处理
    2. 自动重试机制
    3. 速率限制处理
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化收集器

        Args:
            api_key: TikHub API密钥，如果不提供则使用环境变量中的默认值
        """

        self.status = True
        self.api_key = api_key
        self.base_url = settings.TIKHUB_BASE_URL

        if not self.api_key:
            logger.warning("未提供TikHub API密钥，某些功能可能不可用")

        self.headers = {
            'Authorization': f'Bearer {os.getenv("TIKHUB_API_KEY")}',
            'Content-Type': 'application/json'
        }

        self.MAX_RETRIES = 3

    async def _make_request(self, session: ClientSession, url: str, params: Dict) -> Optional[Dict]:
        """通用的API请求处理方法"""
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
                        logger.error(f"获取用户主页 {url} 时出错: HTTP {response.status}, {error_text}")

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
                logger.error(f"获取用户主页时发生网络错误: {str(e)}")
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
                logger.error(f"获取用户主页 {url} 时发生未预期错误: {str(e)}")
                retries += 1
                if retries < self.MAX_RETRIES:
                    await asyncio.sleep(2 ** retries)
                else:
                    raise ExternalAPIError(
                        detail="获取TikTok用户主页时出现未预期错误",
                        service="TikHub",
                        original_error=e
                    )
        # 如果所有重试都失败
        raise ExternalAPIError(
            detail=f"获取用户主页 {url} 时出错: 已达到最大重试次数",
            service="TikHub"
        )


    async def fetch_unique_id(self, url: str, session: ClientSession) -> str:
        """获取用户唯一ID"""
        params = {'url': url}
        request_url = f"{self.base_url}/api/v1/tiktok/web/get_unique_id"
        response = await self._make_request(
            session,
            request_url,
            params,
        )
        return response.get('data', '') if response else ''

    async def fetch_sec_uid(self, url: str, session: ClientSession) -> str:
        """获取用户安全ID"""
        params = {'url': url}
        request_url = f"{self.base_url}/api/v1/tiktok/web/get_sec_user_id"
        response = await self._make_request(
            session,
            request_url,
            params,
        )
        return response.get('data', '') if response else ''

    async def fetch_user_profile(self, url: str) -> Optional[Dict]:
        """
        收集用户档案信息

        Args:
            - url (str): 用户主页URL

        Returns:
            Optional[Dict]: 用户档案数据
        """
        async with ClientSession() as session:
            uniqueId = await self.fetch_unique_id(url, session)
            if not uniqueId:
                raise ValidationError(detail="用户id获取失败", field="url")


            request_url_web = f"{self.base_url}/api/v1/tiktok/web/fetch_user_profile"
            response_web = await self._make_request(
                session,
                request_url_web,
                {'uniqueId': uniqueId},
            )

            request_url_app = f"{self.base_url}/api/v1/tiktok/app/v3/handler_user_profile"
            response_app = await self._make_request(
                session,
                request_url_app,
                {'unique_id': uniqueId},
            )

            return {
                'web_profile': response_web.get('data', {}).get('userInfo', {}),
                'app_profile': response_app.get('data', {}).get('user', {})
            }

    async def fetch_total_fans_count(self, url:str) -> int:
        """
        获取用户总粉丝数

        Args:
            - url (str): 用户主页URL

        Returns:
            int: 用户总粉丝数
        """
        try:
            user_profile = await self.fetch_user_profile(url)
            user_cleaner = UserCleaner()
            user_data = await user_cleaner.clean_user_profile(user_profile)
            return user_data['stats']['followerCount']
        except Exception as e:
            logger.error(f"获取用户粉丝总数失败 - URL: {url}, 错误：{str(e)}")
            return 0

    async def fetch_total_posts_count(self, url: str) -> int:
        """
        获取用户总发布视频数

        Args:
            - url (str): 用户主页URL

        Returns:
            int: 用户总发布视频数
        """
        try:
            user_profile = await self.fetch_user_profile(url)
            user_cleaner = UserCleaner()
            user_data = await user_cleaner.clean_user_profile(user_profile)
            return user_data['stats']['videoCount']
        except Exception as e:
            logger.error(f"获取用户视频总数失败 - URL: {url}, 错误：{str(e)}")
            return 0

    async def stream_user_fans(self, url: str, count: int = 30) -> AsyncGenerator[List[Dict], None]:
        """
        收集用户粉丝数据

        Args:
            - url (str): 用户主页URL（必需）
            - concurrency (int): 并发请求数量（可选，默认2）
            - count (int): 每次请求的粉丝数量（可选，默认30）

        Returns:
            Optional[List[Dict]]: 粉丝数据列表

        Raises:
            ValidationError: 当输入数据无效时
            ExternalAPIError: 当调用外部API出错时
            RateLimitError: 当遇到速率限制时
        """

        total_fans = await self.fetch_total_fans_count(url)
        if total_fans == 0:
            logger.info(f"该用户没有粉丝 - URL: {url}")
            return

        fans = []
        total_collected_fans = 0
        minCursor = 0
        maxCursor = 0
        has_more = True
        logger.info(f"🔍 开始收集用户粉丝数据 - URL: {url}")

        try:
            async with ClientSession() as session:
                secUid = await self.fetch_sec_uid(url, session)

                if not secUid:
                    logger.error(f" 获取用户ID失败，无法收集粉丝数据 - URL: {url}")
                    return

                while has_more and len(fans) < total_fans:
                    params = {
                        'secUid': secUid,
                        'count': count,
                        'maxCursor': maxCursor,
                        'minCursor': minCursor
                    }
                    request_url = f"{self.base_url}/api/v1/tiktok/web/fetch_user_fans"
                    data = await self._make_request(
                        session,
                        request_url,
                        params,
                    )
                    try:
                        data = data['data']
                        new_fans = data['userList']

                        minCursor = data['minCursor']
                        has_more = data['hasMore']
                    except Exception as e:
                        logger.info(f"采集粉丝字段获取问题")
                        continue

                    if new_fans:
                        fans.extend(new_fans)
                        total_collected_fans += len(new_fans)
                        logger.info(f"已收集 {total_collected_fans}/{total_fans} 个粉丝 - URL: {url}")
                        yield new_fans

                    if has_more == False or len(fans) >= total_fans:
                        logger.info(f"✅ 完成收集粉丝数据 - 用户: {url}, 总数: {len(fans)}")
                        break
        except (ValidationError, ExternalAPIError, RateLimitError):
            # 对已知错误直接向上抛出
            raise
        except Exception as e:
            # 捕获所有其他未预期的异常并包装成 ExternalAPIError
            logger.error(f"流式收集评论时发生未预期错误: {str(e)}")
            return

    async def collect_user_posts(self, url: str, count: int = 30) -> AsyncGenerator[List[Dict], None]:
        """
        收集用户发布的视频数据

        Args:
            - url (str): 用户主页URL（必需）

        Returns:
            Optional[List[Dict]]: 视频数据列表
        """
        total_posts = await self.fetch_total_posts_count(url)
        if total_posts == 0:
            logger.info(f"该用户没有发布视频 - URL: {url}")
            return

        maxCursor = 0
        posts = []
        total_collected_posts = 0
        has_more = True
        logger.info(f"开始收集用户发布视频数据 - URL: {url}")

        try:
            async with ClientSession() as session:
                unique_id = await self.fetch_unique_id(url, session)
                if not unique_id:
                    logger.error(f"获取用户ID失败，无法收集视频数据 - URL: {url}")
                    return

                while has_more and len(posts) < total_posts:
                    params = {
                        'unique_id': unique_id,
                        'max_cursor': maxCursor,
                        'count': count,
                        'sort_type': 1
                    }

                    data = await self._make_request(
                        session,
                        f"{self.base_url}/api/v1/tiktok/app/v3/fetch_user_post_videos",
                        params,
                    )

                    data = data['data']
                    new_posts = data['aweme_list']

                    if new_posts:
                        total_collected_posts += len(new_posts)
                        posts.extend(new_posts)
                        logger.info(f"已收集 {total_collected_posts}/{total_posts} 个视频 - 用户: {unique_id}")
                        yield new_posts

                    maxCursor = data['max_cursor']
                    has_more = data['has_more']

                    if has_more == False or len(posts) >= total_posts:
                        logger.info(f"✅ 完成收集发帖数据 - 用户: {url}, 总数: {len(posts)}")
                        break
        except (ValidationError, ExternalAPIError, RateLimitError):
            # 对已知错误直接向上抛出
            raise
        except Exception as e:
            # 捕获所有其他未预期的异常并包装成 ExternalAPIError
            logger.error(f"流式收集用户发布视频数据时发生未预期错误: {str(e)}")
            raise ExternalAPIError(
                detail="流式收集用户发布视频数据时出现未预期错误",
                service="TikHub",
                original_error=e
            )


async def main():
    collector = UserCollector()
    cleaner = UserCleaner()

    url = "https://www.tiktok.com/@galileofarma"

    # 收集用户档案信息
    # profile = await collector.fetch_user_profile(url)
    # cleaned_profile = await cleaner.clean_user_profile(profile)
    # print(json.dumps(cleaned_profile, indent=2))

    # 收集用户粉丝数据
    # async for fans in collector.stream_user_fans(url):
    #     cleaned_fans = await cleaner.clean_user_fans(fans)
    #     print(json.dumps(cleaned_fans, indent=2))
    #     print(f"已收集 {len(cleaned_fans)} 个粉丝")

    # 收集用户发布的视频数据
    async for posts in collector.collect_user_posts(url):
        cleaned_posts = await cleaner.clean_user_posts(posts)
        # print(json.dumps(cleaned_posts, indent=2))
        print(f"已收集 {len(cleaned_posts)} 个视频")


if __name__ == '__main__':
    asyncio.run(main())