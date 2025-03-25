from typing import Dict, Any, List

from app.utils.logger import setup_logger
from app.core.exceptions import ValidationError

# 设置日志记录器
logger = setup_logger(__name__)


class VideoCleaner:
    """抖音视频清洗器，负责处理和标准化原始视频数据"""

    @staticmethod
    async def clean_single_video(video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗和处理单个视频的原始数据

        Args:
            video_data: 原始视频数据

        Returns:
            清洗后的视频数据

        Raises:
            ValidationError: 当输入数据无效时
        """
        if not video_data:
            return {}

        try:
            detail = video_data

            # 处理视频码率（bit_rate）字段
            bit_rate_list = detail.get("video", {}).get("bit_rate", [])
            if bit_rate_list and isinstance(bit_rate_list[0], dict):
                bit_rate = bit_rate_list[0].get("bit_rate", None)
            else:
                bit_rate = None

            # 处理视频链接，取最后一个链接
            video_url_list = detail.get("video", {}).get("play_addr", {}).get("url_list", [])
            video_url = video_url_list[-1] if video_url_list else None

            # 构建清洗后的标准化数据结构
            cleaned_data = {
                # 基础信息
                "aweme_id": detail.get("aweme_id", None),  # 视频ID
                "item_title": detail.get("item_title", None),  # 视频标题
                "desc": detail.get("desc", None),  # 视频描述
                "create_time": detail.get("create_time", None),  # 创建时间

                # 封面图
                "dynamic_cover": detail.get("video", {}).get("dynamic_cover", {}).get("url_list", [None])[0],
                "origin_cover": detail.get("video", {}).get("origin_cover", {}).get("url_list", [None])[0],
                "cover": detail.get("video", {}).get("cover", {}).get("url_list", [None])[0],

                # 视频信息
                "duration": detail.get("video", {}).get("duration", None),  # 视频时长（ms）
                "ratio": detail.get("video", {}).get("ratio", None),  # 视频比例
                "width": detail.get("video", {}).get("width", None),  # 视频宽度
                "height": detail.get("video", {}).get("height", None),  # 视频高度
                "bit_rate": bit_rate,  # 码率
                "video_url": video_url,  # 视频链接

                # 作者信息
                "author": {
                    "uid": detail.get("author", {}).get("uid", None),  # 用户ID
                    "short_id": detail.get("author", {}).get("short_id", None),  # 短ID
                    "nickname": detail.get("author", {}).get("nickname", None),  # 昵称
                    "signature": detail.get("author", {}).get("signature", None),  # 签名
                    "avatar": detail.get("author", {}).get("avatar_larger", {}).get("url_list", [None])[0],
                    "following_count": detail.get("author", {}).get("following_count", None),  # 关注数
                    "follower_count": detail.get("author", {}).get("follower_count", None),  # 粉丝数
                    "favoriting_count": detail.get("author", {}).get("favoriting_count", None),  # 喜欢数
                    "total_favorited": detail.get("author", {}).get("total_favorited", None),  # 获赞数
                    "language": detail.get("author", {}).get("language", None),  # 语言
                    "region": detail.get("author", {}).get("region", None),  # 地区
                },

                # 互动数据
                "statistics": {
                    "comment_count": detail.get("statistics", {}).get("comment_count", None),  # 评论数
                    "digg_count": detail.get("statistics", {}).get("digg_count", None),  # 点赞数
                    "collect_count": detail.get("statistics", {}).get("collect_count", None),  # 收藏数
                    "share_count": detail.get("statistics", {}).get("share_count", None),  # 分享数
                    "download_count": detail.get("statistics", {}).get("download_count", None),  # 下载数
                },

                # 标签信息（text_extra可能不存在或为空列表）
                "tags": " ".join(f"#{tag.get('hashtag_name', '')}" for tag in detail.get("text_extra", [])),

                # 其他信息
                "ocr_content": detail.get("seo_info", {}).get("ocr_content", None),  # OCR文本
                "share_url": detail.get("share_info", {}).get("share_url", None),  # 分享链接
                "music_title": detail.get("music", {}).get("title", None),  # 音乐标题
                "music_author": detail.get("music", {}).get("author", None),  # 音乐作者
                "music_url": detail.get("music", {}).get("play_url", {}).get("url_list", [None])[0],  # 音乐链接

                # 视频权限
                "allow_share": detail.get("video_control", {}).get("allow_share", None),  # 允许分享
                "allow_download": detail.get("video_control", {}).get("allow_download", None),  # 允许下载
                "allow_comment": detail.get("status", {}).get("allow_comment", None),  # 允许评论
                "allow_react": detail.get("video_control", {}).get("allow_react", None),  # 允许react

                # 视频类型标签
                "video_tags": [
                    {
                        "tag_id": tag.get("tag_id", None),
                        "tag_name": tag.get("tag_name", None),
                        "level": tag.get("level", None)
                    }
                    for tag in detail.get("video_tag", [])
                ],
                "video_tags_str": " ".join(tag.get("tag_name", "") for tag in detail.get("video_tag", [])),
            }

            return cleaned_data

        except Exception as e:
            logger.error(f"清洗视频数据时出错: {str(e)}")
            # 返回空字典，不中断流程
            return {}

    async def clean_videos_by_keyword(
            self,
            video_list: List,
            min_digg_count: int = 0
    ) -> List:
        """
        清洗和处理关键词搜索视频列表

        Args:
            video_list: 原始视频列表数据
            min_digg_count: 最小点赞数，过滤点赞数低于此值的视频

        Returns:
            清洗后的视频列表

        Raises:
            ValidationError: 当输入数据无效时
        """
        if not isinstance(video_list, list):
            raise ValidationError(detail="视频数据必须是列表格式", field="video_list")

        # 清洗和处理视频列表
        cleaned_videos = []
        failed_count = 0

        for video in video_list:
            try:
                # 获取视频信息
                video = video['data']['aweme_info']

                # 过滤低点赞数视频
                if video.get('stats', {}).get('diggCount', 0) < min_digg_count:
                    failed_count += 1
                    continue

                # 提取所需信息并构建标准化的视频对象
                cleaned_video = await self.clean_single_video(video)
                cleaned_videos.append(cleaned_video)
            except Exception as e:
                failed_count += 1
                logger.debug(f"处理单个视频时出错: {str(e)}")
                continue

        logger.info(f"已成功清洗 {len(cleaned_videos)} 个关键词视频，失败 {failed_count} 个")
        return cleaned_videos