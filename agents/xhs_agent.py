import asyncio
import datetime
import traceback
from pathlib import Path
from typing import Union

# 重试装饰器
from tenacity import retry, stop_after_attempt
from services.ai_models.chatgpt import ChatGPT
from services.ai_models.claude import Claude
from services.ai_models.whisper import WhisperLemonFox
from app.tools.crawler.douyin import DouYinCrawler
# 导入日志模块
from app.utils.logging_utils import configure_logging
# 导入设置类
from config.settings import Settings


class DyToXhs:
    def __init__(self):
        # 验证环境变量是否已设置
        self._check_environment()
        # 初始化API客户端
        self._initialize_clients()
        # 配置日志记录器 | Configure logger
        self.logger = configure_logging(name=__name__)

    def _check_environment(self) -> None:
        """
        验证所需的环境变量是否已设置。
        :return: None
        """

        # 设置OpenAI API Key
        self.openai_key = Settings.OpenAISettings.API_Key

        # 初始化Claude API Key
        self.claude_api_key = Settings.AnthropicAPISettings.API_Key

        # 验证环境变量是否已设置
        required_env_vars = {
            # OpenAI
            'OPENAI_API_KEY': self.openai_key,

            # Claude
            'ANTHROPIC_API_KEY': self.claude_api_key,
        }
        missing = [k for k, v in required_env_vars.items() if not v]
        if missing:
            self.logger.error(f"Missing required environment variables: {', '.join(missing)}")
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    def _initialize_clients(self) -> None:
        """
        初始化所有API客户端
        :return: None
        """
        self.openai_client = ChatGPT(openai_api_key=self.openai_key)
        self.claude_client = Claude(anthropic_api_key=self.claude_api_key)
        self.douyin_crawler = DouYinCrawler()
        self.wisper = WhisperLemonFox()

    @staticmethod
    def save_chatgpt_to_md(chatgpt_data: dict, output_file: str = None):
        """
        将 ChatGPT API 返回的数据保存为格式化的 Markdown 文件。

        :param chatgpt_data: ChatGPT API 返回的字典数据
        :param output_file: 输出文件路径，如果未提供将基于 aweme_id 动态生成
        """
        try:
            # 提取生成的内容
            # $.message.choices[0].message.content
            content = chatgpt_data.get('message', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

            if not content:
                raise ValueError("未找到内容，请检查返回数据结构。")

            # 确保输出路径有效
            if not output_file:
                output_dir = Path("./results/dy_to_xhs/")
                output_dir.mkdir(parents=True, exist_ok=True)
                aweme_id = chatgpt_data.get('input_data', {}).get('aweme_id', 'unknown')
                output_file = output_dir / f"dy_{aweme_id}.md"
            else:
                output_dir = Path(output_file).parent
                output_dir.mkdir(parents=True, exist_ok=True)

            # 写入 Markdown 文件
            with open(output_file, 'w', encoding='utf-8') as md_file:
                md_file.write(content)

            print(f"Markdown 文件已保存到: {output_file}")
        except Exception as e:
            print(f"保存失败: {e} \n{traceback.format_exc()}")

    def save_claude_to_md(self, claude_data: dict, output_file: str = None):
        """
        将 Claude API 返回的数据保存为格式化的 Markdown 文件。

        :param claude_data: Claude API 返回的字典数据
        :param output_file: 输出文件路径，如果未提供将基于 aweme_id 动态生成
        """
        try:
            # 提取生成的内容（调试信息）
            pre_message = claude_data.get('pre_message', '')

            # 提取生成的内容
            # $.message.content[0].text
            content = claude_data.get('message', {}).get('content', [{}])[0].get('text', '')

            if not content:
                raise ValueError("未找到内容，请检查返回数据结构。")

            # 确保输出路径有效
            if not output_file:
                output_dir = Path("./results/dy_to_xhs/")
                output_dir.mkdir(parents=True, exist_ok=True)
                aweme_id = claude_data.get('input_data', {}).get('douyin_data', {}).get('aweme_id', 'unknown')
                output_file = output_dir / f"dy_{aweme_id}.md"
            else:
                output_dir = Path(output_file).parent
                output_dir.mkdir(parents=True, exist_ok=True)

            # 写入 Markdown 文件
            with open(output_file, 'w', encoding='utf-8') as md_file:
                md_file.write(str(pre_message + "\n" + content))

            self.logger.info(f"Markdown 文件已保存到: {output_file}")
        except Exception as e:
            traceback.format_exc()
            self.logger.error(f"保存失败: {e}")

    @staticmethod
    async def clean_dy_data(data: dict) -> dict:
        """清洗抖音数据，提取有价值的字段，
        当 JSON 中缺失对应 key 时，不会报错，而是返回默认占位值 None

        Args:
            data (dict): 原始响应数据

        Returns:
            dict: 清洗后的数据
        """
        # 获取 detail 数据（若列表为空则使用空字典）
        detail = data["data"]["aweme_details"][0] if data.get("data", {}).get("aweme_details", []) else data["data"][
            "aweme_detail"]

        # 处理视频码率（bit_rate）字段
        bit_rate_list = detail.get("video", {}).get("bit_rate", [])
        if bit_rate_list and isinstance(bit_rate_list[0], dict):
            bit_rate = bit_rate_list[0].get("bit_rate", None)
        else:
            bit_rate = None

        # 处理视频链接，取最后一个链接
        video_url_list = detail.get("video", {}).get("play_addr", {}).get("url_list", [])
        video_url = video_url_list[-1] if video_url_list else None

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
            "allow_react": detail.get("video_control", {}).get("allow_react", None),  # 允许 react

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

    # 抖音视频转录文本
    async def transcriptions(self,
                             file: str,
                             prompt: str = "",
                             response_format: str = "json",
                             language=""
                             ) -> Union[dict, str]:
        response = await self.wisper.transcriptions(
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

        return response

    async def rewrite_douyin_to_xhs(self,
                                    douyin_data: dict,
                                    transcription_data: dict,
                                    output_language: str,
                                    source_platform: str = "抖音",
                                    target_platform: str = "小红书",
                                    target_gender: str = "女性",
                                    target_age: str = "18-30岁",
                                    ) -> dict:
        self.logger.info(f"开始处理抖音数据: {douyin_data.get('item_title', '')}")
        self.logger.info(f"转录文本预览: {transcription_data.get('text', '')[:50]}...")

        # 标题
        item_title = douyin_data.get('item_title', 'N/A')
        # 描述
        desc = douyin_data.get('desc', 'N/A')
        # 封面
        dynamic_cover = douyin_data.get('dynamic_cover', 'N/A')
        # 作者昵称
        nickname = douyin_data.get('author', {}).get('nickname', 'N/A')
        # 作者签名
        signature = douyin_data.get('author', {}).get('signature', 'N/A')
        # 视频评论数
        comment_count = douyin_data.get('statistics', {}).get('comment_count', "N/A")
        # 视频点赞数
        digg_count = douyin_data.get('statistics', {}).get('digg_count', "N/A")
        # 视频收藏数
        collect_count = douyin_data.get('statistics', {}).get('collect_count', "N/A")
        # 视频分享数
        share_count = douyin_data.get('statistics', {}).get('share_count', "N/A")
        # 视频内容标签
        tags = douyin_data.get('tags', 'N/A')
        # 视频内容OCR文本
        ocr_content = douyin_data.get('ocr_content', "N/A")
        # 视频分类标签
        video_tags = douyin_data.get('video_tags_str', "N/A")

        system_prompt = f"""
        你是一位精通{target_platform}算法的全能内容策略师，擅长将任何领域内容转化为{target_gender}用户爱看的爆款笔记。根据输入内容智能匹配创作策略。

        🎯 **目标用户画像**：
        - 性别/人群：{target_gender}（需强化性别相关痛点）
        - 年龄：{target_age if target_age else "18-30岁"}（根据内容自动调整话术）
        - 兴趣标签：{tags if tags else "根据内容智能生成3个以上精准标签"}

        🔥 **跨领域爆款公式**（动态调整）：
        [情绪强度词] + [身份共鸣词] + [场景冲突] + [效果可视化] + [emoji组合]
        ▷ 职场示例：惊呆！会计人快逃｜月底对账到秃头？这个神器10分钟自动生成报表！💼💥
        ▷ 美妆示例：救命！黄黑皮有救｜熬夜脸秒变妈生皮？这粉底液同事追着要链接！💄✨

        ✍️ **智能创作协议**：

        1. **领域自适应标题**：
           - 前5词必带情绪词库（按领域匹配）：
             ✓ 职场/科技：「离谱」「跪了」「打工人必备」
             ✓ 美妆/生活：「绝了」「亲测」「素颜神器」 
             ✓ 教育/母婴：「炸裂」「学霸秘籍」「宝妈偷懒」
           - 身份标签智能生成：
             ■ 分析原文关键词 → 提取「学生党/宝妈/上班族」等标签
             ■ 默认兜底：「{target_age}{target_gender}必看」

        2. **动态标签系统**：
           - 主标签：从内容提取核心名词（如「粉底液」→ #伪素颜神器）
           - 热点标签：结合平台近期热门 hot_hashtags 智能关联
           - 长尾标签：生成「#早八快速出门妆」等具体场景标签
           ▶ 生成规则：核心关键词 + 痛点 / 效果 + 神器 / 大法

        3. **跨领域正文框架**：
           ■ 痛点挖掘（根据内容类型）：
             ✓ 职场：加班/效率/汇报痛点 →「谁懂做PPT到凌晨的痛啊😭」
             ✓ 美妆：脱妆/肤色/上妆难度 →「黄黑皮真的不配拥有伪素颜吗？」
           ■ 解决方案场景化：
             ✓ 用「三步法」「10秒口诀」替代技术术语
             ✓ 对比描述：「之前...现在...」句式使用率≥1次
           ■ 跨领域互动模板：
             ✓ 美妆类：「@闺蜜团来测评」「左滑查看妆效对比」
             ✓ 职场类：「评论区蹲模板」「带薪摸鱼小技巧」

        4. **智能纠错与本土化**：
           - 建立多领域术语替换库：
             ▷ 科技类：Agent→小助理｜SaaS→神器
             ▷ 美妆类：持妆→不脱妆｜妆效→妈生感
           - 发音纠错增强：同时检查中英文近音词（如「遮瑕」vs「遮霞」）

        5. **视觉策略升级**：
           - 根据领域生成图片指令：
             ✓ 职场：前后对比图（电脑屏幕+时间对比）
             ✓ 美妆：效果九宫格（素颜→淡妆→浓妆）
           - 重点数据可视化：用「⬆️300%效率」替代「大幅提升」

        🎯 **输出格式**：
        ▌标题：[情绪词][智能身份标签][场景痛点][效果承诺][emoji]
        ▌正文：包含领域适配的互动埋点+场景对比
        ▌标签组：3-5个（自动生成领域精准标签+热点标签）
        ▌图片描述：符合领域特征的视觉指令
        """

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

        try:
            model = "claude-3-5-sonnet-20241022"  # 请根据实际情况选择模型
            message = await self.claude_client.chat(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=3000,
                timeout=60,
            )

            pre_message = (
                f"> Debug Info\n"
                f" - 来源平台: {source_platform}，目标平台: {target_platform}\n"
                f" - 使用模型: {model}\n"
                f" - 生成时间: {datetime.datetime.now()}\n"
                f" - 系统提示词: {system_prompt}\n"
                f" - 用户提示词: {user_prompt}\n\n"
            )

            result = {
                "pre_message": pre_message,
                "message": message,
                "output_language": output_language,
                "input_data": {
                    "douyin_data": douyin_data,
                    "transcription_data": transcription_data,
                },
            }
            return result
        except Exception as e:
            self.logger.error(f"Error in rewriting Douyin content: {e}")
            raise e

    # 使用示例
    async def main(self):
        item_url = "https://v.douyin.com/tuhfcH-R6jM/"
        # 获取抖音数据
        douyin_data = await self.douyin_crawler.fetch_one_video_by_share_url_v2(item_url)

        # 清洗抖音数据
        douyin_data = await self.clean_dy_data(douyin_data)

        print(douyin_data)

        # 获取抖音视频链接
        video_url = douyin_data.get("video_url")

        # 抖音视频描述
        desc = douyin_data.get("desc")

        # 抖音数据转文本
        transcription_data = await self.transcriptions(
            file=video_url,
            prompt=desc
        )

        print(transcription_data)

        # 定义来源平台
        source_platform = "抖音"
        # 定义目标平台
        target_platform = "小红书"
        # 定义输出语言
        output_language = "zh"

        target_gender = "女性"

        target_age = "18-30岁"

        # 使用 Claude 重写内容
        rewrite_data = await self.rewrite_douyin_to_xhs(
            douyin_data,
            transcription_data,
            output_language,
            source_platform,
            target_platform,
            target_gender,
            target_age,
        )

        print(rewrite_data)

        # 保存Claude生成的内容
        self.save_claude_to_md(rewrite_data)


if __name__ == "__main__":
    DyToXhs = DyToXhs()

    asyncio.run(DyToXhs.main())