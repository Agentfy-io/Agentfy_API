import datetime
import json
import re
import time
from typing import Union, Dict, Any, Optional, AsyncGenerator

# 导入日志模块
from app.utils.logger import setup_logger
from app.core.exceptions import ValidationError, ExternalAPIError, InternalServerError

from services.ai_models.chatgpt import ChatGPT
from services.ai_models.claude import Claude
from services.ai_models.whisper import WhisperLemonFox
from services.crawler.douyin.video_crawler import DouYinCrawler
from services.cleaner.douyin.video_cleaner import VideoCleaner

# Set up logger
logger = setup_logger(__name__)

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


class XHSAgent:
    """抖音内容转小红书的工具类，提供视频数据转换、转录和内容改写功能"""

    def __init__(self, tikhub_api_key: Optional[str] = None, openai_api_key: Optional[str] = None,
                 claude_api_key: Optional[str] = None,
                 lemon_fox_api_key: Optional[str] = None):
        """
        初始化抖音到小红书转换工具

        Args:
            api_key: API密钥，如果不提供则使用环境变量中的默认值
        """
        # 初始化AI模型客户端
        self.chatgpt = ChatGPT(openai_api_key=openai_api_key)
        self.claude = Claude( anthropic_api_key=claude_api_key)
        self.whisper = WhisperLemonFox(lemon_fox_api_key=lemon_fox_api_key)
        self.video_crawler = DouYinCrawler(tikhub_api_key)
        self.video_cleaner = VideoCleaner()

        # 保存API配置
        self.api_key = tikhub_api_key

        # 如果没有提供API密钥，记录警告
        if not self.api_key:
            logger.warning("未提供API密钥，某些功能可能不可用")

        # 设置日志记录器
        self.logger = setup_logger(__name__)

    async def fetch_source_video_data(self, item_url: str) -> Dict[str, Any]:
        """
        从抖音API获取视频数据

        Args:
            item_url: 抖音视频分享链接

        Returns:
            视频数据
        """
        try:
            logger.info(f"开始获取抖音视频数据: {item_url}")

            # 获取抖音数据
            douyin_data_raw = await self.video_crawler.fetch_one_video_by_share_url(item_url)

            # 清洗抖音数据
            douyin_data = await self.video_cleaner.clean_single_video(douyin_data_raw)

            logger.info(f"抖音视频数据获取完成")
            return douyin_data
        except Exception as e:
            logger.error(f"获取抖音视频数据失败: {str(e)}")
            raise ExternalAPIError(f"获取抖音视频数据失败: {str(e)}")

    async def transcriptions(
            self,
            file: str,
            prompt: str = "",
            response_format: str = "json",
            language: str = ""
    ) -> Union[dict, str]:
        """
        抖音视频转录文本

        Args:
            file: 视频文件路径或URL
            prompt: 提示词
            response_format: 响应格式
            language: 语言

        Returns:
            转录结果
        """
        try:
            logger.info(f"开始转录视频: {file}")

            response = await self.whisper.transcriptions(
                file=file,
                response_format=response_format,
                speaker_labels=False,
                prompt=prompt,
                language=language,
                callback_url="",
                translate=False,
                timestamp_granularities=None,
                timeout=60
            )

            logger.info(f"视频转录完成")
            return response

        except Exception as e:
            logger.error(f"视频转录失败: {str(e)}")
            raise ExternalAPIError(f"视频转录失败: {str(e)}")

    async def rewrite_douyin_to_xhs(
            self,
            douyin_data: Dict[str, Any],
            transcription_data: Dict[str, Any],
            output_language: str,
            source_platform: str = "抖音",
            target_platform: str = "小红书",
            target_gender: str = "女性",
            target_age: str = "18-30岁"
    ) -> Dict[str, Any]:
        """
        将抖音内容重写为小红书风格

        Args:
            douyin_data: 抖音视频数据
            transcription_data: 视频转录数据
            output_language: 输出语言
            source_platform: 源平台
            target_platform: 目标平台
            target_gender: 目标性别
            target_age: 目标年龄段

        Returns:
            改写后的内容
        """
        start_time = time.time()

        try:
            logger.info(f"Starting to process Douyin data: {douyin_data.get('item_title', '')}")
            logger.info(f"Transcription text preview: {transcription_data.get('text', '')[:50]}...")

            # Extract Douyin data fields
            item_title = douyin_data.get('item_title', 'N/A')
            desc = douyin_data.get('desc', 'N/A')
            dynamic_cover = douyin_data.get('dynamic_cover', 'N/A')
            nickname = douyin_data.get('author', {}).get('nickname', 'N/A')
            signature = douyin_data.get('author', {}).get('signature', 'N/A')
            comment_count = douyin_data.get('statistics', {}).get('comment_count', "N/A")
            digg_count = douyin_data.get('statistics', {}).get('digg_count', "N/A")
            collect_count = douyin_data.get('statistics', {}).get('collect_count', "N/A")
            share_count = douyin_data.get('statistics', {}).get('share_count', "N/A")
            tags = douyin_data.get('tags', 'N/A')
            ocr_content = douyin_data.get('ocr_content', "N/A")
            video_tags = douyin_data.get('video_tags_str', "N/A")

            # Build system prompt
            system_prompt = f"""
            你是一位精通 {target_platform} 算法的内容策略专家，擅长将任意领域内容转化为 {target_gender} 用户爱看的爆款笔记。请根据输入内容，智能匹配平台传播策略与内容风格，生成更具吸引力的内容结构。

            🎯 **用户画像设定**：
            - 性别/人群：{target_gender}（强化性别相关痛点）
            - 年龄段：{target_age if target_age else "18-30岁"}（根据内容自动调整话术风格）
            - 兴趣标签：{tags if tags else "智能提取内容关键词生成3个以上精准兴趣标签"}

            🔍 **内容结构创作公式**：
            [情绪词] + [身份标签] + [场景冲突] + [可视化效果] + [emoji]
            示例：
            - 职场类：跪了！打工人快抄｜月底报表一键生成太爽了💻📈
            - 美妆类：救命！黄黑皮也能妈生感｜这款粉底液真的神💄✨

            📌 **创作策略要点**：

            1️⃣ **爆款标题生成规则**：
            - 前5个字包含情绪词（按领域匹配）：
              - 职场/科技：离谱、跪了、打工人必备、全网疯传
              - 美妆/生活：亲测、救命、素颜神器、熬夜急救
              - 教育/母婴：炸裂、学霸秘籍、宝妈必看
            - 身份标签智能提取：学生党、宝妈、社畜、打工人（如无，默认输出"{target_age}{target_gender}必看"）

            2️⃣ **正文内容结构建议**：
            - 痛点切入句（领域相关）：「谁懂加班到凌晨的崩溃😩」
            - 场景对比句：「之前...现在...」句式 ≥ 1次
            - 场景化解决方案描述（避免术语，建议使用"三步法"、"10秒口诀"）
            - 跨领域互动提示：
              - 美妆：「@姐妹测评」「左滑看对比」
              - 职场：「评论区求模板」「偷偷用太香了」

            3️⃣ **标签生成逻辑**：
            - 主标签：从内容中提取核心词（如「AI助理」→ #效率神器）
            - 热点标签：结合平台热搜词自动关联
            - 场景标签：关键词 + 痛点/效果 + 工具/大法，例如：#早八拯救神器、#打工人必备工具

            4️⃣ **图片指令建议**：
            - 职场类：前后对比图（左：手动表格+崩溃脸，右：AI生成+轻松脸）
            - 美妆类：妆效九宫格（素颜→底妆→完成）
            - 强调"免费""效率""神器"等关键词，使用视觉高亮（如红色字体/emoji）

            📤 **最终输出格式**：
            请严格输出以下四个字段组成的 Python 字典（Dictionary）：

            "title": "【情绪词】【身份标签】【场景痛点】【效果承诺】【emoji组合】",
            "content": "正文内容，包含场景痛点描述、场景对比、解决方案、互动引导等。",
            "hashtags": ["#标签1", "#标签2", "#标签3", "#标签4", "#标签5"],
            "image_desc": "图像内容说明，适配该领域内容的可视化指令"

            请不要输出解释文字，只输出字典内容。 
            """

            # Build user prompt
            user_prompt = f"""
            **输入信息：**
            - 标题：{item_title}
            - 描述：{desc}
            - 视频封面：{dynamic_cover}
            - 作者信息：{nickname}（{signature}）
            - 点赞数：{digg_count}
            - 收藏数：{collect_count}
            - 评论数：{comment_count}
            - 分享数：{share_count}
            - 标签：{tags}
            - OCR文本：{ocr_content}
            - 分类标签：{video_tags}
            - 转录文本：{transcription_data['text']}
            - 语言：{output_language}
            """

            # Use Claude for content rewriting
            model = "claude-3-5-sonnet-20241022"
            message = await self.claude.chat(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=3000,
                timeout=60,
            )

            # 构建调试信息
            pre_message = (
                f"> Debug Info\n"
                f" - 来源平台: {source_platform}，目标平台: {target_platform}\n"
                f" - 使用模型: {model}\n"
                f" - 生成时间: {datetime.datetime.now()}\n"
                f" - 系统提示词: {system_prompt}\n"
                f" - 用户提示词: {user_prompt}\n\n"
            )

            # 计算处理时间
            processing_time = time.time() - start_time
            content = message['choices'][0]['message']['content']

            # 处理返回的JSON格式（可能包含在Markdown代码块中）
            content = re.sub(
                r"```json\n|\n```|```|\n",
                "",
                content.strip()
            )

            content = json.loads(content)

            # 构建结果
            result = {
                "note": content,
                "output_language": output_language,
                "input_data": {
                    "douyin_data": douyin_data,
                    "transcription_data": transcription_data,
                },
                "metadata": {
                    "cost": message.get('cost'),
                    "source_platform": source_platform,
                    "target_platform": target_platform,
                    "target_gender": target_gender,
                    "target_age": target_age,
                    "generated_at": datetime.datetime.now().isoformat(),
                    "processing_time": processing_time
                }
            }

            logger.info(f"内容改写完成，耗时: {processing_time:.2f}秒")

            return result

        except ExternalAPIError as e:
            # 直接向上传递API错误
            raise
        except Exception as e:
            logger.error(f"内容改写时发生未预期错误: {str(e)}")
            raise InternalServerError(f"内容改写时发生未预期错误: {str(e)}")

    async def url_to_xhs(
            self,
            item_url: str,
            source_platform: str = "抖音",
            target_platform: str = "小红书",
            target_gender: str = "女性",
            target_age: str = "18-30岁"
    ) -> Dict[str, Any]:
        """
        将单个抖音视频转换为小红书风格

        Args:
            item_url: 抖音视频分享链接
            source_platform: 源平台
            target_platform: 目标平台
            target_gender: 目标性别
            target_age: 目标年龄段

        Returns:
            转换结果
        """
        start_time = time.time()
        try:

            # 获取抖音数据
            douyin_data = await self.fetch_source_video_data(item_url)

            # 获取抖音视频链接和描述
            video_url = douyin_data.get("video_url")
            desc = douyin_data.get("desc")

            # 抖音视频转录
            transcription_data = await self.transcriptions(
                file=video_url,
                prompt=desc
            )

            rewrite_data = await self.rewrite_douyin_to_xhs(
                douyin_data,
                transcription_data,
                output_language="zh",
                source_platform=source_platform,
                target_platform=target_platform,
                target_gender=target_gender,
                target_age=target_age
            )

            return rewrite_data
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            raise InternalServerError(f"处理失败: {str(e)}")

    async def keyword_to_xhs(
            self,
            keyword: str,
            source_platform: str = "抖音",
            target_platform: str = "小红书",
            target_gender: str = "女性",
            target_age: str = "18-30岁"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        将抖音关键词搜索结果转换为小红书风格

        Args:
            keyword: 搜索关键词
            source_platform: 源平台
            target_platform: 目标平台
            target_gender: 目标性别
            target_age: 目标年龄段

        Yields
            转换结果
        """
        # 验证输入参数
        if not keyword or not isinstance(keyword, str):
            raise ValidationError(detail="搜索关键词不能为空", field="keyword")

        video_list = []  # 临时存储分析结果
        content = ""  # 生成的内容

        try:
            # 获取抖音视频搜索结果
            async for video in self.video_crawler.stream_video_search_results(keyword):
                # 获取视频数据
                video_data = await self.video_cleaner.clean_videos_by_keyword(video)
                video_list.extend(video_data)
                # 获取视频链接和描述
                yield {
                    "keyword": keyword,
                    "is_complete": False,
                    "message": f"正在采集相关内容，已获取 {len(video_list)} 条视频数据，正在处理...",
                    "total_collected": len(video_list),
                    "content": content
                }

            if not video_list:
                yield {
                    "keyword": keyword,
                    "is_complete": False,
                    "message": "未找到相关视频，请尝试其他关键词, 或耐心等待接口恢复",
                    "total_collected": 0,
                    "content": content
                }
                return

            sorted_list = sorted(video_list, key=lambda x: x.get('statistics', {}).get('digg_count', 0), reverse=True)
            top_video = sorted_list[0]
            print(top_video)

            # 获取抖音视频链接和描述
            video_url = top_video.get("video_url")
            desc = top_video.get("desc")
            print(video_url, desc)

            # 抖音视频转录
            transcription_data = await self.transcriptions(
                file=video_url,
                prompt=desc
            )

            # 使用Claude重写内容
            rewrite_data = await self.rewrite_douyin_to_xhs(
                top_video,
                transcription_data,
                output_language="zh",
                source_platform=source_platform,
                target_platform=target_platform,
                target_gender=target_gender,
                target_age=target_age
            )

            yield {
                "keyword": keyword,
                "is_complete": True,
                "message": "已从搜索结果中提取最佳视频，内容改写完成",
                "total_collected": len(video_list),
                "content": rewrite_data
            }
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            raise InternalServerError(f"处理失败: {str(e)}")