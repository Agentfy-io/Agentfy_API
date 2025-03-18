import json
import os
import re
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import asyncio
from app.config import settings
from services.ai_models.chatgpt import ChatGPT
from services.ai_models.claude import Claude
from typing import Dict, Any, List, Optional, AsyncGenerator
from dotenv import load_dotenv
from app.utils.logger import logger
from services.crawler.user_crawler import UserCollector
from services.cleaner.user_cleaner import UserCleaner

# 加载环境变量
load_dotenv()


class UserAgent:
    """用户/达人分析器
    基础用户数据展示 （粉丝数、点赞数、评论数、转发数、作品数、关注数、获赞数、获赞率、评论率、转发率、互动率,商业
    用户发布作品统计 (作品类型、作品风格、平均点赞数、平均评论数、平均转发数、平均互动数，平均作品时长，平均播放量， 商业视频数量，总作品数)
    用户发布作品趋势数据 （作品发布时间、作品发布频率、作品发布时长、作品发布互动、作品发布变现）
    粉丝画像分析 （粉丝性别、粉丝年龄、粉丝地域、粉丝兴趣、粉丝活跃度、粉丝互动度、粉丝变现度）
    热门视频数据 （热门视频标题、热门视频内容、热门视频互动、热门视频变现）
    标签和相似达人
    """


    def __init__(self, tikhub_api_key: Optional[str] = None):
        """
        初始化达人分析器

        Args:
            tikhub_api_key: TikHub API密钥
        """
        self.total_fans = 0
        self.total_posts = 0

        # 初始化 ChatGPT 和 Claude
        self.chatgpt = ChatGPT()
        self.claude = Claude()

        # 初始化 UserCollector 和 UserCleaner
        self.user_collector = UserCollector(tikhub_api_key)
        self.user_cleaner = UserCleaner()

        # 保存TikHub API配置
        self.tikhub_api_key = tikhub_api_key
        self.tikhub_base_url = settings.TIKHUB_BASE_URL

        # 如果没有提供TikHub API密钥，记录警告
        if not self.tikhub_api_key:
            logger.warning("未提供TikHub API密钥，某些功能可能不可用")

        # 支持的分析类型列表
        self.analysis_types = ['profile_analysis', 'post_analysis']

        # 加载系统提示
        self._load_system_prompts()

    def _load_system_prompts(self):
        self.system_prompts = {
            "profile_analysis": """
            You are a data analyst specializing in social media analytics. You have been tasked with analyzing a TikTok user profile to provide insights and recommendations for growth. Please create a detailed report following this structure:
            Please create a detailed report following this structure:

            Profile Overview（Create a summary table of key profile information ）

            Engagement Analysis(Present a table of metrics using stats and metrics data)

            business Profile Analysis (if applicable)

            contact information

            account settings and features and identification

            overall account analysis 

            Analyze:

            Content Volume (based on video count)
            Commercial Integration (commerce features, business links)
            Platform Utilization (cross-platform presence, bio links)
            Content Categories and Focus Areas


            Business Profile Analysis (if applicable)

            Examine:

            Business Category
            Commercial Features
            App Presence (iOS/Android links)
            Business Contact Information
            Commercial Integration Level


            Growth Opportunities and Recommendations

            Provide 3-5 actionable recommendations based on:

            Current performance metrics
            Platform feature utilization
            Engagement patterns
            Business integration opportunities


            Please format the tables using markdown and provide clear, concise insights for each section. Include percentages and comparative metrics where relevant to add context to the analysis.
            Additional Guidelines:

            Use clear section headers with markdown formatting
            Present data in well-organized tables
            Include calculated metrics like engagement rates
            Highlight notable strengths and areas for improvement
            Keep the tone professional but accessible
            Include emojis in each section for better readability

            Please generate a comprehensive report that would be valuable for both the account owner and social media managers.
                    """,
            "post_analysis": """你是一位专业的数据分析师，专门为社交媒体营销和TikTok影响者数据分析提供深度解读。你的任务是基于提供的TikTok网红统计数据，生成一份完整的 报告，该报告应包含 关键数据摘要、可视化趋势图、内容分析及结论建议，并符合以下内容要求：
            # TikTok 影响者数据分析报告

            📌 1️⃣ Markdown 结构
            - 标题层级清晰，以 `#` 作为标题标记，确保内容清晰可读。
            - 数据表格（Tables） 以 `|` 分隔，表头使用 `|---|---|` 进行格式化，便于展示账号的核心数据。
            - 适当使用 `**加粗**` 和 `-` 进行列表划分，确保层次分明，信息直观。

            ---

            ## 📊 2️⃣ 可视化图表
            你需要根据 `post_trend` 生成以下六个图表，并返回 **Markdown 形式的图片链接（或 Base64 图片）**：

            ### **📅 每日发帖数趋势（折线图）**
            - **描述**: 展示每日发帖数量的变化趋势。
            - **数据源**: `post_trend.post_trend` (x 轴: 日期, y 轴: 发帖数)
            - **图表示例**:
              ![每日发帖趋势](<图片URL>)

            ### **👍 每日点赞趋势（折线图）**
            - **描述**: 反映用户点赞的增长趋势。
            - **数据源**: `interaction_trend.digg_count` (x 轴: 日期, y 轴: 点赞数)
            - **图表示例**:
              ![每日点赞趋势](<图片URL>)

            ### **💬 每日评论趋势（折线图）**
            - **描述**: 观察每日评论数量的变化，判断受众互动活跃度。
            - **数据源**: `interaction_trend.comment_count` (x 轴: 日期, y 轴: 评论数)
            - **图表示例**:
              ![每日评论趋势](<图片URL>)

            ### **🔄 每日分享趋势（折线图）**
            - **描述**: 追踪每日分享次数，评估内容的传播能力。
            - **数据源**: `interaction_trend.share_count` (x 轴: 日期, y 轴: 分享数)
            - **图表示例**:
              ![每日分享趋势](<图片URL>)

            ### **▶️ 每日播放趋势（折线图）**
            - **描述**: 展示每日播放量的波动情况，评估视频的整体表现。
            - **数据源**: `interaction_trend.play_count` (x 轴: 日期, y 轴: 播放数)
            - **图表示例**:
              ![每日播放趋势](<图片URL>)

            ### **📊 视频时长分布（饼图）**
            - **描述**: 统计视频时长分布，分析观众更喜欢的内容长度。
            - **数据源**: `post_duration_distribution`
            - **图表示例**:
              ![视频时长分布](<图片URL>)

            ---

            ## 📊 3️⃣ 账号数据概览

            ### **核心统计数据**
            | 统计项 | 数值 |
            |---|---|
            | **帖子总数** | X |
            | **总点赞数** | X |
            | **总评论数** | X |
            | **总分享数** | X |
            | **总播放数** | X |
            | **总下载数** | X |
            | **总AI生成视频数** | X |
            | **总VR视频数** | X |
            | **总广告视频数** | X |

            ### **计算指标**
            | 统计项 | 数值 |
            |---|---|
            | **平均点赞数** | X |
            | **平均评论数** | X |
            | **平均分享数** | X |
            | **平均播放数** | X |
            | **最高点赞数**（日期X） | X |
            | **最高评论数**（日期X） | X |
            | **最高分享数**（日期X） | X |
            | **最高播放数**（日期X） | X |
            | **总互动量**（点赞 + 评论 + 分享） | X |
            | **点赞率** | X% |
            | **评论率** | X% |
            | **分享率** | X% |
            | **播放转化率** | X% |

            ---

            ## 🔍 4️⃣ 关键趋势分析

            ### **📈 账号增长趋势**
            - 过去 7 天的 **平均发帖数**，与长期趋势对比，判断是否增长或下降。
            - 点赞、评论、分享、播放的 **增长率**，评估账号受欢迎程度的变化。
            - **最近30天内互动最高的帖子**：
              - **发布时间**
              - **视频内容**
              - **热门话题**
              - **成功关键因素**
            - **是否存在增长瓶颈**（如点赞率下降、播放量减少等）。

            ### **🎯 内容互动分析**
            - **点赞最高的帖子**（分析发布时间、内容类型）
            - **评论最多的帖子**（是否引发讨论、争议）
            - **分享最多的帖子**（是否具有病毒传播特性）
            - **最高播放量帖子**（分析视频质量、音乐、封面、标题等）

            ### **🕒 最佳发布时间分析**
            - **计算平均发布时间的黄金时段**
            - **分析工作日 vs. 周末 的互动差异**
            - **找出增长最快的时间点**

            ### **⏳ 视频时长表现**
            - 统计 **不同时长区间的平均互动率**
            - 评估 **短视频 vs. 长视频 哪种效果更好**
            - 观众更喜欢的 **时长（如15-30s 是否表现最佳）**

            ### **🏷️ 热门标签分析**
            - **Top 5 hashtags 使用频率**
            - **评估标签对互动的影响**
            - **哪些标签带来更多流量**
            - **推荐使用高互动标签**
            - **图表示例**：
              ![热门标签使用频率](<图片URL>)

            ---

            ## 🎯 5️⃣ 结论与优化建议

            ### **🕒 最佳发布时间**
            - 建议在 **`X 时段`** 发布内容，以最大化曝光率和互动率。

            ### **🔥 热门标签建议**
            - 例如 **`#farmacia #farma #cuidadodelapiel #skincare`** 可能提高曝光。
            - 是否调整标签策略，如结合更多趋势标签（**#healthtips #beautytips**）。

            ### **🚀 短视频 vs. 长视频优化策略**
            - 如果 **短视频（15-30s）表现最佳**，建议保持该策略。
            - 如果 **长视频（60s+）互动较差**，建议减少发布频率或调整内容策略。

            ### **📈 账号趋势评估**
            - **当前账号是增长还是下降趋势？**
            - **如果上升，如何维持增长？如果下降，如何优化？**
            - **互动数据是否健康？点赞多但评论、分享少？如何优化？**

            ### **📢 未来优化策略**
            - **内容创新**
            - **用户互动**
            - **跨平台引流**
            ---



            """
        }

    """---------------------------------------------通用方法/工具类方法---------------------------------------------"""
    async def generate_analysis_report(self, uniqueId: str, analysis_type: str, data: Dict[str, Any]) -> str:
        """
        生成报告并转换为HTML

        Args:
            uniqueId (str): 用户ID
            analysis_type (str): 分析类型
            data (Dict[str, Any]): 分析数据

        Returns:
            str: HTML报告的本地文件URL
        """
        if analysis_type not in self.system_prompts:
            raise ValueError(f"Invalid report type: {analysis_type}. Choose from {self.system_prompts.keys()}")

        try:
            # 获取系统提示
            sys_prompt = self.system_prompts[analysis_type]

            # 获取用户提示
            user_prompt = f"Generate a report for the {analysis_type} analysis based on the following data:\n{json.dumps(data, ensure_ascii=False)}, for user {uniqueId}"

            # 生成报告
            response = await self.chatgpt.chat(
                system_prompt=sys_prompt,
                user_prompt=user_prompt
            )

            report = response["choices"][0]["message"]["content"].strip()

            # 保存Markdown报告
            report_dir = "reports"
            os.makedirs(report_dir, exist_ok=True)

            report_md_path = os.path.join(report_dir, f"report_{uniqueId}.md")
            with open(report_md_path, "w", encoding="utf-8") as f:
                f.write(report)

            # 转换为HTML
            html_content = self.convert_markdown_to_html(report, f"{analysis_type.title()} Analysis for {uniqueId}")
            html_filename = f"report_{uniqueId}.html"
            html_path = os.path.join(report_dir, html_filename)

            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # 生成本地文件URL
            absolute_path = os.path.abspath(html_path)

            # 构建file://协议URL
            file_url = f"file://{absolute_path}"

            # 确保路径分隔符是URL兼容的
            if os.name == 'nt':  # Windows系统
                # Windows路径需要转换为URL格式
                file_url = file_url.replace('\\', '/')

            return file_url
        except Exception as e:
            logger.error(f"生成报告时发生错误: {str(e)}")
            raise

    def convert_markdown_to_html(self, markdown_content: str, title: str = "Analysis Report") -> str:
        """
        将Markdown内容转换为HTML

        Args:
            markdown_content (str): Markdown内容
            title (str): HTML页面标题

        Returns:
            str: HTML内容
        """
        try:
            import markdown
        except ImportError:
            print("请安装markdown库: pip install markdown")
            return f"<pre>{markdown_content}</pre>"

        # 转换Markdown为HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['tables', 'fenced_code', 'codehilite', 'toc']
        )

        # 创建完整HTML文档
        css = """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }
        h1, h2, h3 { margin-top: 1.5em; color: #111; }
        pre { background-color: #f6f8fa; border-radius: 3px; padding: 16px; overflow: auto; }
        code { font-family: SFMono-Regular, Consolas, Menlo, monospace; background-color: #f6f8fa; padding: 0.2em 0.4em; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f6f8fa; }
        """

        html_document = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>{css}</style>
    </head>
    <body>
        <h1>{title}</h1>
        {html_content}
    </body>
    </html>
        """

        return html_document

    """---------------------------------------------用户/达人基础信息分析方法---------------------------------------------"""
    async def fetch_user_profile_analysis(self, url: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        分析用户/达人的基础信息
        """
        start_time = time.time()
        if not url or not re.match(r"https://(www\.)?tiktok\.com/@[\w\.-]+", url):
            raise ValueError("Invalid TikTok user profile URL")

        try:
            yield {
                "user_profile_url": url,
                "is_complete": False,
                "message": '正在采集用户/达人{}的基础信息...请耐心等待'.format(url),
                "uniqueId": '',
                "analysis_report": '',
                "profile_raw_data": {},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time": round(time.time() - start_time, 2)
            }

            data = await self.user_collector.fetch_user_profile(url)
            data = await self.user_cleaner.clean_user_profile(data)

            uniqueId = data['accountIdentifiers']['uniqueId']

            logger.info("正在分析用户/达人基础信息...")

            yield {
                "user_profile_url": url,
                "is_complete": False,
                "message": f"已完成用户/达人{url}的信息采集， 正在生成分析报告...",
                "uniqueId": '',
                "analysis_report": '',
                "profile_raw_data": {},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time": round(time.time() - start_time, 2)
            }

            report_url = await self.generate_analysis_report(uniqueId, 'profile_analysis', data)

            yield{
                "user_profile_url": url,
                "is_complete": True,
                "message": f"已完成用户/达人{url}的基础信息分析，报告已生成",
                "uniqueId": uniqueId,
                "profile_raw_data": data,
                "analysis_report": report_url,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time": round(time.time() - start_time, 2)
            }
        except Exception as e:
            logger.error(f"分析用户/达人基础信息时发生错误: {str(e)}")
            yield {
                "user_profile_url": url,
                "is_complete": False,
                'error': str(e),
                'message': f"分析用户/达人{url}基础信息时发生错误: {str(e)}",
                "uniqueId": '',
                "profile_raw_data": {},
                "analysis_report": '',
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time": round(time.time() - start_time, 2)
            }

    asyncio.run(main())

