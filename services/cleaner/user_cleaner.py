import os
import time
import json
from typing import Dict, List, Any
import pandas as pd
import aiofiles
import asyncio
from aiohttp import ClientSession

from app.config import settings
from app.utils.logger import setup_logger
from app.core.exceptions import ValidationError

# 设置日志记录器
logger = setup_logger(__name__)


class UserCleaner:
    """
    TikTok数据清洗, 包括用户档案数据、粉丝数据、帖子数据
    """

    def __init__(self):
        """初始化清洗器，设置函数映射"""
        self.status = True


    async def clean_user_profile(self, user_profile: Dict[str,Any]) -> Dict:
        """
        清洗用户档案数据并转换为JSON格式的DataFrame

        Args:
            user_profile (Dict[str,Any]): 用户档案数据

        Returns:
            Dict: 整合后的清洗数据
        """
        try:
            logger.info("开始清洗用户档案数据")

            profile_web = user_profile.get('web_profile', {})
            profile_app = user_profile.get('app_profile', {})

            # 获取 Web 端和 App 端用户数据
            web_user = profile_web.get('user', {})
            web_stats = profile_web.get('stats', {})

            # 避免零除错误
            heart_count = int(web_stats.get('heartCount', 0)) or 1
            video_count = int(web_stats.get('videoCount', 0)) or 1
            follower_count = int(profile_app.get('follower_count', 0)) or 1

            # 账户类型映射
            account_type_mapping = {
                1: "Personal",
                2: "Creator",
                3: "Business"
            }
            account_type = account_type_mapping.get(profile_app.get('account_type'), "Unknown")

            # 构建数据字典
            cleaned_data = {
                "accountIdentifiers": {
                    "uniqueId": web_user.get('uniqueId', ''),
                    "uid": web_user.get('id', ''),
                    "secUid": web_user.get('secUid', ''),
                    "accountType": account_type,
                },
                "profile": {
                    "nickname": web_user.get('nickname', ''),
                    "avatarUrl": web_user.get('avatarLarger', ''),
                    "signature": web_user.get('signature', ''),
                    "bioLink": web_user.get('bioLink', {}).get('link', ''),
                    "category": web_user.get('commerceUserInfo', {}).get('category', ''),
                    "region": web_user.get('region', ''),
                    "language": web_user.get('language', ''),
                    "isStar": bool(profile_app.get('is_star', False)),
                    "isEffectArtist": bool(profile_app.get('is_effect_artist', False)),
                },
                "stats": {
                    "heartCount": heart_count,
                    "videoCount": video_count,
                    "friendCount": int(web_stats.get('friendCount', 0)),
                    "followerCount": follower_count,
                    "followingCount": int(profile_app.get('following_count', 0)),
                    "totalFavorited": int(profile_app.get('total_favorited', 0)),
                    "visibleVideos": int(profile_app.get('visible_videos_count', 0)),
                },
                "metrics": {
                    "engagementRate": round((follower_count / heart_count) * 100, 2),
                    "avgLikesPerVideo": round(heart_count / video_count, 2),
                    "avgLikesPerFollower": round(heart_count / follower_count, 2),
                },
                "business": {
                    "companyName": profile_app.get('biz_account_info', {}).get('rba_user_info', {}).get('company_name',
                                                                                                        ''),
                    "isCommerceUser": bool(web_user.get('commerceUserInfo', {}).get('commerceUser', False)),
                    "androidLink": web_user.get('commerceUserInfo', {}).get('downLoadLink', {}).get('android', ''),
                    "iosLink": web_user.get('commerceUserInfo', {}).get('downLoadLink', {}).get('ios', ''),
                    "isAdVirtual": bool(web_user.get('isADVirtual', False)),
                    "isTtSeller": bool(web_user.get('ttSeller', False)),
                    "commerceLevel": int(profile_app.get('commerce_user_level', 0)),
                },
                "contact": {
                    "email": profile_app.get('bio_email', ''),
                    "youtubeChannel": profile_app.get('youtube_channel_id', ''),
                    "twitterId": profile_app.get('twitter_id', ''),
                    "shareUrl": profile_app.get('share_info', {}).get('share_url', ''),
                },
                "settings": {
                    "isVerified": bool(web_user.get('verified', False)),
                    "isPrivate": bool(web_user.get('privateAccount', False)),
                    "followingVisibility": web_user.get('followingVisibility', ''),
                    "embedPermission": bool(web_user.get('profileEmbedPermission', False)),
                }
            }

            logger.info("用户档案数据清洗完成")
            return cleaned_data

        except Exception as e:
            logger.error(f"清洗用户档案数据时发生错误: {str(e)}")
            raise

    async def clean_user_fans(self, fans_data: List[Dict]) -> List[Dict]:
        """
        清洗粉丝数据

        Args:
            fans_data: 粉丝数据

        Returns:
            pd.DataFrame: 清洗后的粉丝数据
        """

        try:
            logger.info(f"🔍 开始清洗粉丝数据")
            # 转换为DataFrame
            df = pd.json_normalize(fans_data)

            # 提取所需字段
            result = pd.DataFrame()
            result['uid'] = df['user.id']
            result['unique_id'] = df['user.uniqueId']
            result['nickname'] = df['user.nickname']
            result['avatarLarger'] = df['user.avatarLarger']
            result['signature'] = df['user.signature']
            result['secUid'] = df['user.secUid']

            # 转换为Dict
            result = result.to_dict(orient='records')

            logger.info(f"✅ 清洗粉丝数据完成")

            return result

        except Exception as e:
            logger.error(f"❌ 清洗粉丝数据时发生错误: {str(e)}")
            raise

    async def clean_user_posts(self, posts_data: List[Dict]) -> List[Dict]:
        """
        清洗用户帖子数据

        Args:
            posts_data: 用户帖子数据

        Returns:
            pd.DataFrame: 清洗后的用户帖子数据
        """

        try:
            # 转换为DataFrame
            df = pd.json_normalize(posts_data)

            # 处理特殊字段
            def parse_anchors_extra(row):
                try:
                    if pd.isna(row):
                        return {'is_ec_video': False}
                    data = json.loads(row) if isinstance(row, str) else row
                    return data
                except:
                    return {'is_ec_video': False}

            def parse_cha_list(row):
                if row is None:
                    return {}
                return {
                    item['cha_name']: item['cid']
                    for item in row
                }

            def parse_content_desc_extra(row):
                result = {}
                if row is None:
                    return result
                for item in row:
                    if item.get('sec_uid') is None:
                        result[item['hashtag_name']] = item.get('hashtag_id', '')
                return result

            df['anchors_extras'] = df['anchors_extras'].apply(parse_anchors_extra)
            df['is_ec_video'] = df['anchors_extras'].apply(lambda x: x.get('is_ec_video', False))
            df['cha_list'] = df['cha_list'].apply(parse_cha_list)
            df['content_desc_extra'] = df['content_desc_extra'].apply(parse_content_desc_extra)

            # 构建结果DataFrame
            result = pd.DataFrame()

            # 基本信息
            result['created_by_ai'] = df['aigc_info.created_by_ai'].fillna(False)
            result['is_ec_video'] = df['is_ec_video']
            result['sec_uid'] = df['author.sec_uid']
            result['unique_id'] = df['author.unique_id']
            result['aweme_id'] = df['aweme_id']
            result['cha_list'] = df['cha_list'].apply(json.dumps)
            result['hashtags'] = df['content_desc_extra'].apply(json.dumps)
            result['content_type'] = df['content_type'].fillna('')
            result['create_time'] = df['create_time'].fillna(0)

            # 基础字段
            # is_nff_or_nr false = 该内容可以出现在feed流和推荐中
            basic_fields = ['desc', 'desc_language', 'group_id', 'has_danmaku', 'has_promote_entry',
                            'has_vs_entry', 'is_ads', 'is_nff_or_nr', 'is_pgcshow', 'is_relieve',
                            'is_top', 'is_vr', 'item_duet', 'item_react', 'item_stitch']
            for field in basic_fields:
                result[field] = df[field].fillna('')

            # 统计数据
            stat_fields = ['collect_count', 'comment_count', 'digg_count', 'download_count',
                           'play_count', 'share_count', 'whatsapp_share_count']
            for field in stat_fields:
                result[field] = df[f'statistics.{field}'].fillna(0)

            # 状态信息
            status_fields = ['in_reviewing', 'is_delete', 'is_prohibited', 'reviewed']
            for field in status_fields:
                result[field] = df[f'status.{field}'].fillna(False)

            # 视频信息
            result['download_addr'] = df['video.download_no_watermark_addr.url_list'].apply(
                lambda x: x[2] if isinstance(x, list) and len(x) > 2 else '')
            result['duration'] = df['video.duration'].fillna(0)
            result['allow_download'] = df['video_control.allow_download'].fillna(False)

            # 转换为Dict
            result = result.to_dict(orient='records')


            logger.info(f"清洗用户帖子数据完成")

            return result

        except Exception as e:
            logger.error(f"清洗用户帖子数据时发生错误: {str(e)}")
            return {}


async def main():
    """Example usage of the UserCleaner"""
    cleaner = UserCleaner()

    # Example data
    test_data = {
        'web_profile': {
            'user': {
                'id': '123',
                'uniqueId': 'testuser',
                'secUid': 'user123',
                'nickname': 'Test User',
                'avatarLarger': 'https://example.com/avatar.jpg',
                'signature': 'Test signature',
                'bioLink': {
                    'link': 'https://example.com'
                },
                'commerceUserInfo': {
                    'category': 'test',
                },
                'region': 'US',
                'language': 'en',
            },
            'stats': {
                'heartCount': 100,
                'videoCount': 10,
                'friendCount': 50,
                    'followerCount': 1000,
                    'followingCount': 500,
                    'totalFavorited': 1000,
                    'visibleVideos': 10,
            }
        },
        'app_profile': {
            'follower_count': 1000,
            'account_type': 2,
            'is_star': False,
            'is_effect_artist': False,
            'following_count': 500,
            'total_favorited': 1000,
            'visible_videos_count': 10,
            'commerce_user_level': 1,
            'youtube_channel_id': 'test',
            'twitter_id': 'test',
            'share_info': {
                'share_url': 'https://example.com'
            }
        }
    }

if __name__ == '__main__':
    asyncio.run(main())
