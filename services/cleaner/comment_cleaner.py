from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

from app.utils.logger import setup_logger
from app.core.exceptions import ValidationError

# 设置日志记录器
logger = setup_logger(__name__)


class CommentCleaner:
    """TikTok评论清洗器，负责处理和标准化原始评论数据"""

    async def clean_video_comments(
            self,
            comments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        清洗和处理特定视频的原始评论

        Args:
            aweme_id: 视频ID
            comments: 原始评论列表

        Returns:
            清洗后的评论列表

        Raises:
            ValidationError: 当输入数据无效时
        """
        aweme_id = comments.get('aweme_id', '')
        if not aweme_id:
            raise ValidationError(detail="视频ID不能为空", field="aweme_id")

        comments = comments.get('comments', [])
        if not isinstance(comments, list):
            raise ValidationError(detail="评论数据必须是列表", field="comments")

        try:
            # 清洗和处理评论
            cleaned_comments = []
            for comment in comments:
                if not isinstance(comment, dict):
                    logger.warning(f"跳过无效评论格式: {type(comment)}")
                    continue

                # 提取用户数据，防止KeyError
                user_data = comment.get('user', {}) or {}

                # 创建清洗后的评论对象
                cleaned_comment = {
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
                    logger.warning(f"跳过无效评论: 缺少ID或文本")

            logger.info(f"成功清洗视频 {aweme_id} 的 {len(cleaned_comments)} 条评论")
            return {
                'aweme_id': aweme_id,
                'comments': cleaned_comments,
                'total_comments': len(cleaned_comments),
            }

        except Exception as e:
            logger.error(f"清洗视频 {aweme_id} 评论时出错: {str(e)}")
            # 返回已清洗的评论（可能是部分），而不是抛出异常中断整个流程
            return []

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