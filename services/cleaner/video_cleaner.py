# data_cleaning.py
import os
from typing import Dict, Any, List, Tuple

import pandas as pd
import json
from pathlib import Path
import asyncio
import aiofiles

import
from utils.logger import logger


class VideoCleaner:

    async def clean_one_video(self, video_data:Dict) -> Dict[str, Any]:
        """
        Clean and process raw video data for a specific video

        Parameters:
        - data (Dict[str, Any]): Required. Raw video data containing 'aweme_id' and 'video_info'
        """

        if not video_data:
            raise ValueError("data parameter is required")
        try:
            logger.info(f"Cleaning video aweme_id: {video_data.get('aweme_id', '')}")
            # Clean and process video data
            cha_list = video_data.get('cha_list', [])
            content_desc_extra = video_data.get('content_desc_extra', {})
            cha_list_cleaned = hashtags = []
            if cha_list:
                cha_list_cleaned = [{'cid': cha.get('cid', ''), 'title': cha.get('cha_name', '')} for cha in cha_list]
            if content_desc_extra:
                hashtags = [{'hashtag_id': tag.get('id', ''), 'name': tag.get('name', '')} for tag in content_desc_extra]

            # remove empty values
            hashtags = [tag for tag in hashtags if tag['name']]

            cleaned_video = {
                'music':{
                    'id': video_data.get("added_sound_music_info", {}).get('mid', ''),
                    'title': video_data.get("added_sound_music_info", {}).get('title', ''),
                    'owner_id': video_data.get("added_sound_music_info", {}).get('owner_id', ''),
                    'owner_nickname': video_data.get("added_sound_music_info", {}).get('owner_nickname', ''),
                    'play_url': video_data.get("added_sound_music_info", {}).get('play_url', {}).get('uri', []),
                },
                'created_by_ai': video_data.get('aigc_info', {}).get('created_by_ai', False),
                'author':{
                    'avatar': video_data.get('author', {}).get('avatar_larger', {}).get('url_list', [])[0],
                    'sec_uid': video_data.get('author', {}).get('sec_uid', ''),
                    'nickname': video_data.get('author', {}).get('nickname', ''),
                    'unique_id': video_data.get('author', {}).get('unique_id', ''),
                    'uid': video_data.get('author', {}).get('uid', ''),
                    'youtube_channel_id': video_data.get('author', {}).get('youtube_channel_id', ''),
                    'youtube_channel_title': video_data.get('author', {}).get('youtube_channel_title', ''),
                    'ins_id': video_data.get('author', {}).get('ins_id', ''),
                    'twitter_id': video_data.get('author', {}).get('twitter_id', ''),
                    'twitter_name': video_data.get('author', {}).get('twitter_name', ''),
                    'region': video_data.get('author', {}).get('region', ''),
                },
                'aweme_id': video_data.get('aweme_id', ''),
                'cha_list': cha_list_cleaned,
                'hashtags': hashtags,
                'content_type': video_data.get('content_type', ''),
                'desc': video_data.get('desc', ''),
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
                'statistics':{
                    'collect_count': video_data.get('statistics', {}).get('collect_count', 0),
                    'comment_count': video_data.get('statistics', {}).get('comment_count', 0),
                    'digg_count': video_data.get('statistics', {}).get('digg_count', 0),
                    'download_count': video_data.get('statistics', {}).get('download_count', 0),
                    'play_count': video_data.get('statistics', {}).get('play_count', 0),
                    'share_count': video_data.get('statistics', {}).get('share_count', 0),
                },
                'is_reviewing': video_data.get('status', {}).get('is_reviewing', False),
                'is_prohibited': video_data.get('status', {}).get('is_prohibited', False),
                'is_delete': video_data.get('status', {}).get('is_delete', False),
                'reviewed': video_data.get('status', {}).get('reviewed', False),
                'play_address': video_data.get('video', {}).get('play_addr', {}).get('url_list', [])[0],
                'duration': video_data.get('video', {}).get('duration', 0),
                'allow_download': video_data.get('video', {}).get('allow_download', False),
            }

            return cleaned_video
        except Exception as e:
            logger.error(f"Error cleaning video data: {str(e)}")
            raise

    async def clean_hashtag_videos(self, video_data:Dict) -> List[Dict[str, Any]]:
        """
        Clean and process raw video data for a specific hashtag

        Parameters:
        - data (Dict[str, Any]): Required. Raw video data containing 'chi_id' and 'video_list'
        """

        if not video_data:
            raise ValueError("data parameter is required")
        try:
            # Clean and process video data
            cleaned_videos = []
            for video in video_data:
                cleaned_video = await self.clean_one_video(data=video)
                cleaned_videos.append(cleaned_video)
            logger.info(f"Cleaned {len(cleaned_videos)} videos for hashtag")
            return cleaned_videos
        except Exception as e:
            logger.error(f"Error cleaning hashtag video data: {str(e)}")
            raise

    async def clean_keyword_videos(self, video_data:Dict) -> List[Dict[str, Any]]:
        """
        Clean and process raw video data for a specific keyword

        Parameters:
        - data (Dict[str, Any]): Required. Raw video data containing 'keyword' and 'video_list'
        """

        if not video_data:
            raise ValueError("data parameter is required")
        try:
            # Clean and process video data
            cleaned_videos = []
            for video in video_data:
                video = video.get('aweme_info', {})
                if not video:
                    logger.error("Invalid video data format")
                    continue
                cleaned_video = await self.clean_one_video(data=video)
                cleaned_videos.append(cleaned_video)
            return cleaned_videos
        except Exception as e:
            logger.error(f"Error cleaning keyword video data: {str(e)}")
            raise

if __name__ == '__main__':
    # load json data, uf8
    with open('../../results/video_test/video_data.json', 'r', encoding='utf-8') as f:
        video_data = json.load(f)

    with open('../../results/video_test/hashtag_videos.json', 'r', encoding='utf-8') as f:
        hashtag_data = json.load(f)

    with open('../../results/video_test/keyword_videos.json', 'r', encoding='utf-8') as f:
        keyword_data = json.load(f)

    cleaner = VideoCleaner()
    cleaned_video = asyncio.run(cleaner.clean_one_video(data=video_data))
    cleaned_hashtag_videos = asyncio.run(cleaner.clean_hashtag_videos(data=hashtag_data))
    cleaned_keyword_videos = asyncio.run(cleaner.clean_keyword_videos(data=keyword_data))

    #save cleaned data, ut8
    with open('../../results/video_test/cleaned_video.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_video, f, ensure_ascii=False, indent=4)

    with open('../../results/video_test/cleaned_hashtag_videos.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_hashtag_videos, f, ensure_ascii=False, indent=4)

    with open('../../results/video_test/cleaned_keyword_videos.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_keyword_videos, f, ensure_ascii=False, indent=4)