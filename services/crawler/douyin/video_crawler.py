import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
import aiohttp
from aiohttp import ClientSession
from dotenv import load_dotenv

from app.config import settings
from app.utils.logger import setup_logger
from app.core.exceptions import ExternalAPIError, ValidationError, RateLimitError
from services.cleaner.douyin.video_cleaner import VideoCleaner

# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()


class DouYinCrawler:
    """抖音数据爬虫，负责从TikHub API获取抖音视频数据"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化抖音爬虫

        Args:
            api_key: TikHub API密钥，如果不提供则使用环境变量中的默认值
        """
        self.api_key = api_key or settings.TIKHUB_API_KEY
        self.base_url = settings.TIKHUB_BASE_URL.rstrip('/')

        if not self.api_key:
            logger.warning("未提供TikHub API密钥")
            raise RuntimeError("缺少TikHub API密钥")

        self.headers = {
            "User-Agent": "Agentfy.io/1.0.0",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def make_request(
            self,
            endpoint: str,
            params: Dict[str, Any],
            session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """
        通用请求方法，用于调用TikHub API

        Args:
            endpoint: API端点路径
            params: 请求参数
            session: aiohttp会话

        Returns:
            API响应数据

        Raises:
            ExternalAPIError: 当调用外部API出错时
        """
        url = f"{self.base_url}{endpoint}"

        try:
            logger.info(f"正在请求: {url} 参数: {params}")
            async with session.get(url, params=params, headers=self.headers) as response:
                response_data = await response.json()

                # 检查响应code
                if response_data.get("code") != 200:
                    raise ExternalAPIError(
                        detail=f"TikHub API返回错误: code={response_data.get('code')}",
                        service="TikHub",
                        status_code=response.status
                    )

                return response_data
        except Exception as e:
            logger.error(f"API请求失败: {str(e)}")
            raise ExternalAPIError(
                detail=f"请求TikHub API时出现错误: {str(e)}",
                service="TikHub",
                original_error=e
            )

    async def fetch_one_video_by_share_url(self, item_url: str) -> Dict[str, Any]:
        """
        获取单一抖音视频数据，使用 TikHub APP V3 API

        Args:
            item_url: 抖音视频分享链接

        Returns:
            抖音视频数据

        Raises:
            ExternalAPIError: 当调用外部API出错时
        """
        endpoint = "/api/v1/douyin/app/v3/fetch_one_video_by_share_url"
        params = {
            'share_url': item_url
        }

        try:
            async with aiohttp.ClientSession() as session:
                result = await self.make_request(endpoint, params, session)
                video_data = result['data']['aweme_detail']
                if not video_data:
                    raise ExternalAPIError(detail="未找到视频数据", service="TikHub")

                return video_data
        except ExternalAPIError:
            raise
        except Exception as e:
            # 捕获所有其他未预期的异常并包装成 ExternalAPIError
            logger.error(f"获取视频数据时出错: {str(e)}")
            raise ExternalAPIError(detail=f"获取视频数据时出错: {str(e)}")

    async def stream_video_search_results(
            self,
            keyword: str,
            sort_type: str = "_1",
            publish_time: str = "_0",
            filter_duration: str = "_0",
            page: int = 1
    ) -> AsyncGenerator[List[Dict], None]:
        """
        获取指定关键词的视频搜索结果V2

        Args:
            keyword (str): 搜索关键词（必需）
            sort_type (str): 排序类型（可选，默认"_1"最多点赞）
                - "_0": 综合(General)
                - "_1": 最多点赞(More likes)
                - "_2": 最新发布(New)
            publish_time (str): 发布时间筛选（可选，默认"_0"不限）
                - "_0": 不限(No Limit)
                - "_1": 一天之内(last 1 day)
                - "_7": 一周之内(last 1 week)
                - "_180": 半年之内(last half year)
            filter_duration (str): 视频时长筛选（可选，默认"_0"不限）
                - "_0": 不限(No Limit)
                - "_1": 1分钟以下(1 minute and below)
                - "_2": 1-5分钟(1-5 minutes)
                - "_3": 5分钟以上(5 minutes more)
            page (int): 页码（可选，默认1）

        Returns:
            AsyncGenerator[List[Dict], None]: 视频搜索结果数据流

        Raises:
            ValidationError: 当输入数据无效时
            ExternalAPIError: 当调用外部API出错时
            RateLimitError: 当遇到速率限制时
        """
        logger.info(f"开始搜索视频 - 关键词: {keyword}")
        request_url = f"/api/v1/douyin/app/v3/fetch_video_search_result_v2"

        try:
            current_page = page
            search_id = ""
            has_more = True
            total_videos_collected = 0

            async with ClientSession() as session:
                while has_more:
                    params = {
                        'keyword': keyword,
                        'sort_type': sort_type,
                        'publish_time': publish_time,
                        'filter_duration': filter_duration,
                        'page': current_page
                    }
                    # 仅在search_id不为空时添加到参数中
                    if search_id:
                        params['search_id'] = search_id

                    logger.info(f"请求第 {current_page} 页 - 关键词: {keyword}")
                    data = await self.make_request(
                        request_url,
                        params,
                        session,
                    )

                    try:
                        result_data = data['data']
                        # 获取下一页需要的search_id
                        search_id = result_data.get('business_config', {}).get('next_page', {}).get('search_id', '')
                        videos = result_data.get('business_data', [])
                        has_more = result_data.get('business_config', {}).get('has_more', False)

                        if videos:
                            total_videos_collected += len(videos)
                            logger.info(
                                f"第 {current_page} 页获取到 {len(videos)} 个视频，总计: {total_videos_collected}")
                            yield videos
                        else:
                            logger.info(f"第 {current_page} 页未找到视频，结束搜索")
                            has_more = False

                        current_page += 1
                    except Exception as e:
                        logger.error(f"解析搜索结果数据时出错: {str(e)}")
                        raise ExternalAPIError(f"解析搜索结果数据时出错: {str(e)}")

                    # 休眠1秒以避免速率限制
                    await asyncio.sleep(1)

                logger.info(
                    f"✅ 完成搜索视频 - 关键词: {keyword}, 总页数: {current_page - 1}, 总视频数: {total_videos_collected}")

        except (ValidationError, ExternalAPIError, RateLimitError):
            # 当遇到验证错误、API错误或速率限制时，返回空列表
            yield []
        except Exception as e:
            # 捕获所有其他未预期的异常，记录错误并返回空列表
            logger.error(f"视频搜索过程中发生未预期错误: {str(e)}")
            yield []

    async def main(self):
        """
        主函数示例，展示如何使用此类的主要方法

        Returns:
            无返回值，仅作为示例
        """
        # 示例1：获取单个视频数据
        item_url = "https://v.douyin.com/e3x2fjE/"
        video_data = await self.fetch_one_video_by_share_url(item_url)
        print("单个视频数据:", video_data)

        # 示例2：搜索视频
        keyword = "中华娘"
        video_cleaner = VideoCleaner()

        async for videos in self.stream_video_search_results(keyword, page=1):
            cleaned_video = await video_cleaner.clean_videos_by_keyword(videos)
            print("搜索到视频数量:", len(cleaned_video))


if __name__ == '__main__':
    # 创建抖音爬虫对象
    dy_crawler = DouYinCrawler()
    # 运行主函数
    asyncio.run(dy_crawler.main())