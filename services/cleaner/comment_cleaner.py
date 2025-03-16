from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

import pandas as pd

from app.utils.logger import setup_logger
from app.core.exceptions import ValidationError

# 设置日志记录器
logger = setup_logger(__name__)


class CommentCleaner:
    """TikTok评论清洗器，负责处理和标准化原始评论数据"""

    def __init__(self):
        self.status = True

    async def clean_video_comments(
            self,
            comments: List[Dict]
    ) -> List[Dict]:
        """
        清洗和处理特定视频的原始评论

        Args:
            comments: 原始评论列表

        Returns:
            清洗后的评论列表

        Raises:
            ValidationError: 当输入数据无效时
        """
        if self.status:

            if not isinstance(comments, list):
                raise ValidationError(detail="评论数据必须是列表", field="comments")

            # 清洗和处理评论
            cleaned_comments = []
            aweme_id = None

            try:
                for comment in comments:
                    # 如果该条 comment 的数据不符合预期，就直接跳过（如不是 dict 或为空）
                    if not isinstance(comment, dict) or not comment:
                        logger.warning(f"跳过无效评论格式: {type(comment)}")
                        continue

                    try:
                        # 这里如果 'user' 或 'aweme_id' 等关键字段不存在，就会触发 KeyError
                        user_data = comment['user']
                        aweme_id = comment['aweme_id']

                        # 创建清洗后的评论对象
                        cleaned_comment = {
                            'aweme_id': comment.get('aweme_id'),
                            'comment_id': comment.get('cid', ''),
                            'text': self._clean_text(comment.get('text', '')),
                            'comment_language': comment.get('comment_language', ''),
                            'digg_count': self._parse_int(comment.get('digg_count', 0)),
                            'reply_count': self._parse_int(comment.get('reply_comment_total', 0)),
                            'commenter_secuid': user_data.get('sec_uid', ''),
                            'commenter_uniqueId': user_data.get('unique_id', ''),
                            'commenter_region': user_data.get('region', ''),
                            'ins_id': user_data.get('ins_id', ''),
                            'twitter_id': user_data.get('twitter_id', ''),
                            'create_time': comment.get('create_time', '')
                        }

                        # 只添加有效评论（必须有ID和文本）
                        if cleaned_comment['comment_id'] and cleaned_comment['text']:
                            cleaned_comments.append(cleaned_comment)
                        else:
                            logger.warning("跳过无效评论: 缺少ID或文本")

                    except KeyError as e:
                        # 当出现 KeyError 时，记录错误并返回已成功清洗的评论（不抛出异常）
                        logger.error(f"评论数据缺少关键字段: {str(e)}，跳过处理")
                        continue

                # 如果所有评论都处理成功，此时再进行去重，并返回
                logger.info(f"成功清洗视频 {aweme_id} 的 {len(cleaned_comments)} 条评论")
                return cleaned_comments

            except Exception as e:
                logger.error(f"处理视频{aweme_id}时出现异常: {str(e)}，返回已处理的评论")
                return cleaned_comments


    def _clean_text(self, text: str) -> str:
        """清洗评论文本，去除多余空白"""
        if not isinstance(text, str):
            return ""
        return text.strip()

    def _parse_int(self, value: Any) -> int:
        """安全解析整数值"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0