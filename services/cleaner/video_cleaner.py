from typing import Dict, Any, List, Optional
import asyncio

from app.utils.logger import setup_logger
from app.core.exceptions import ValidationError

# 设置日志记录器
logger = setup_logger(__name__)


class VideoCleaner:
    """TikTok视频清洗器，负责处理和标准化原始视频数据"""

    async def clean_single_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
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
            raise ValidationError(detail="视频数据不能为空", field="video_data")

        aweme_id = video_data.get('aweme_id', '')
        if not aweme_id:
            raise ValidationError(detail="视频ID不能为空", field="aweme_id")

        try:
            logger.info(f"清洗视频 {aweme_id}")

            # 处理标签列表
            cha_list = video_data.get('cha_list', [])
            content_desc_extra = video_data.get('content_desc_extra', {})
            cha_list_cleaned = []
            hashtags = []

            if cha_list:
                cha_list_cleaned = [{'cid': cha.get('cid', ''), 'title': cha.get('cha_name', '')} for cha in cha_list]

            if content_desc_extra:
                hashtags = [{'hashtag_id': tag.get('id', ''), 'name': tag.get('name', '')} for tag in
                            content_desc_extra]

            # 移除空值
            hashtags = [tag for tag in hashtags if tag['name']]

            # 创建清洗后的视频对象
            music_info = video_data.get("added_sound_music_info", {}) or {}
            author_info = video_data.get('author', {}) or {}
            statistics = video_data.get('statistics', {}) or {}
            status = video_data.get('status', {}) or {}
            video_info = video_data.get('video', {}) or {}

            cleaned_video = {
                'music': {
                    'id': music_info.get('mid', ''),
                    'title': music_info.get('title', ''),
                    'owner_id': music_info.get('owner_id', ''),
                    'owner_nickname': music_info.get('owner_nickname', ''),
                    'play_url': self._get_first_item(music_info.get('play_url', {}).get('uri', [])),
                },
                'created_by_ai': video_data.get('aigc_info', {}).get('created_by_ai', False),
                'author': {
                    'avatar': self._get_first_item(author_info.get('avatar_larger', {}).get('url_list', [])),
                    'sec_uid': author_info.get('sec_uid', ''),
                    'nickname': author_info.get('nickname', ''),
                    'unique_id': author_info.get('unique_id', ''),
                    'uid': author_info.get('uid', ''),
                    'youtube_channel_id': author_info.get('youtube_channel_id', ''),
                    'youtube_channel_title': author_info.get('youtube_channel_title', ''),
                    'ins_id': author_info.get('ins_id', ''),
                    'twitter_id': author_info.get('twitter_id', ''),
                    'twitter_name': author_info.get('twitter_name', ''),
                    'region': author_info.get('region', ''),
                },
                'aweme_id': aweme_id,
                'cha_list': cha_list_cleaned,
                'hashtags': hashtags,
                'content_type': video_data.get('content_type', ''),
                'desc': self._clean_text(video_data.get('desc', '')),
                'create_time': video_data.get('create_time', ''),
                'group_id': video_data.get('group_id', ''),
                'has_vs_entry': video_data.get('has_vs_entry', False),
                'is_ads': video_data.get('is_ads', False),
                'is_nff_or_nr': video_data.get('is_nff_or_nr', False),
                'is_pgcshow': video_data.get('is_pgcshow', False),
                'is_top': video_data.get('is_top', False),
                'is_vr': video_data.get('is_vr', False),
                'item_duet': video_data.get('item_duet', False),
                'item_react': video_data.get('item_react', False),
                'item_stitch': video_data.get('item_stitch', False),
                'region': video_data.get('region', ''),
                'share_url': video_data.get('share_info', {}).get('share_url', ''),
                'share_desc': video_data.get('share_info', {}).get('share_desc', ''),
                'statistics': {
                    'collect_count': self._parse_int(statistics.get('collect_count', 0)),
                    'comment_count': self._parse_int(statistics.get('comment_count', 0)),
                    'digg_count': self._parse_int(statistics.get('digg_count', 0)),
                    'download_count': self._parse_int(statistics.get('download_count', 0)),
                    'play_count': self._parse_int(statistics.get('play_count', 0)),
                    'share_count': self._parse_int(statistics.get('share_count', 0)),
                },
                'is_reviewing': status.get('is_reviewing', False),
                'is_prohibited': status.get('is_prohibited', False),
                'is_delete': status.get('is_delete', False),
                'reviewed': status.get('reviewed', False),
                'play_address': self._get_first_item(video_info.get('play_addr', {}).get('url_list', [])),
                'duration': self._parse_int(video_info.get('duration', 0)),
                'allow_download': video_info.get('allow_download', False),
            }

            return {
                'aweme_id': aweme_id,
                'video': cleaned_video,
            }

        except ValidationError:
            # 直接向上传递验证错误
            raise
        except Exception as e:
            logger.error(f"清洗视频 {aweme_id} 数据时出错: {str(e)}")
            # 返回基本数据，不中断流程
            return {
                'aweme_id': aweme_id,
                'error': str(e),
            }

    async def clean_videos_by_hashtag(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗和处理话题标签视频列表

        Args:
            video_list: 原始视频列表数据

        Returns:
            清洗后的视频列表

        Raises:
            ValidationError: 当输入数据无效时
        """
        if not isinstance(video_data, dict):
            raise ValidationError(detail="视频数据必须是字典格式", field="video_data")
        video_list = video_data.get('videos', [])
        try:
            # 清洗和处理视频列表
            cleaned_videos = []
            failed_count = 0

            for video in video_list:
                try:
                    cleaned_video = await self.clean_app_video(video)
                    if cleaned_video:
                        cleaned_videos.append(cleaned_video)
                except Exception as e:
                    logger.error(f"清洗话题视频时出错: {str(e)}")
                    failed_count += 1
                    continue

            logger.info(f"已成功清洗 {len(cleaned_videos)} 个话题视频，失败 {failed_count} 个")
            return {
                'hashtag': video_data.get('chi_id', ''),
                'videos': cleaned_videos,
                'video_count': len(cleaned_videos),
            }

        except ValidationError:
            # 直接向上传递验证错误
            raise
        except Exception as e:
            logger.error(f"清洗话题视频列表时出错: {str(e)}")
            # 返回已清洗的视频（可能是部分），而不是抛出异常中断整个流程
            return {
                'videos': [],
                'error': str(e),
            }

    async def clean_videos_by_keyword(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗和处理关键词搜索视频列表

        Args:
            video_data: 原始视频列表数据

        Returns:
            清洗后的视频列表

        Raises:
            ValidationError: 当输入数据无效时
        """
        if not isinstance(video_data, dict):
            raise ValidationError(detail="视频数据必须是字典格式", field="video_data")

        video_list = video_data.get('videos', [])

        try:
            # 清洗和处理视频列表
            cleaned_videos = []
            failed_count = 0


            for video in video_list:
                try:
                    cleaned_video = {
                        'aweme_id': video.get('id', ''),
                        'desc': video.get('desc', ''),
                        'create_time': video.get('create_time', ''),
                        'playAddr': video.get('video', {}).get('playAddr', ''),
                        'duration': video.get('video', {}).get('duration', 0),
                        'uid': video.get('author', {}).get('id', ''),
                        'uniqueId': video.get('author', {}).get('uniqueId', ''),
                        'nickname': video.get('author', {}).get('nickname', ''),
                        'avatarMedium': video.get('author', {}).get('avatarMedium', ''),
                        'signature': video.get('author', {}).get('signature', ''),
                        'secUid': video.get('author', {}).get('secUid', ''),
                        'privateAccount': video.get('author', {}).get('privateAccount', False),
                        'mid': video.get('music', {}).get('id', ''),
                        'musicTitle': video.get('music', {}).get('title', ''),
                        'musicAuthor': video.get('music', {}).get('authorName', ''),
                        'album': video.get('music', {}).get('album', ''),
                        'diggCount': video.get('stats', {}).get('diggCount', 0),
                        'shareCount': video.get('stats', {}).get('shareCount', 0),
                        'commentCount': video.get('stats', {}).get('commentCount', 0),
                        'playCount': video.get('stats', {}).get('playCount', 0),
                        'collectCount': video.get('stats', {}).get('collectCount', 0),
                        'author_following_count': video.get('authorStats', {}).get('followingCount', 0),
                        'author_follower_count': video.get('authorStats', {}).get('followerCount', 0),
                        'author_heart_count': video.get('authorStats', {}).get('heartCount', 0),
                        'author_video_count': video.get('authorStats', {}).get('videoCount', 0),
                        'author_heart': video.get('authorStats', {}).get('heart', 0),
                        'author_digg_count': video.get('authorStats', {}).get('diggCount', 0),
                        'isAds': video.get('isAds', False),

                    }
                    cleaned_videos.append(cleaned_video)
                except Exception as e:
                    logger.error(f"清洗关键词视频时出错: {str(e)}")
                    failed_count += 1
                    continue

            logger.info(f"已成功清洗 {len(cleaned_videos)} 个关键词视频，失败 {failed_count} 个")
            return {
                'keyword': video_data.get('keyword', ''),
                'videos': cleaned_videos,
                'video_count': len(cleaned_videos),
            }

        except ValidationError:
            # 直接向上传递验证错误
            raise
        except Exception as e:
            logger.error(f"清洗关键词视频列表时出错: {str(e)}")
            # 返回已清洗的视频（可能是部分），而不是抛出异常中断整个流程
            return {
                'videos': [],
                'error': str(e),
            }

    def _clean_text(self, text: str) -> str:
        """清洗文本，去除多余空白"""
        if not isinstance(text, str):
            return ""
        return text.strip()

    def _parse_int(self, value: Any) -> int:
        """安全解析整数值"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    def _get_first_item(self, list_data: List) -> Any:
        """安全获取列表的第一个元素"""
        if isinstance(list_data, list) and list_data:
            return list_data[0]
        return ""