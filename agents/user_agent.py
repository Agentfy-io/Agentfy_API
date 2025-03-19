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
            """,
            "post_stats_analysis": """
                # System Prompt: TikTok Analytics Report Generator        
                You are an expert data analyst specializing in social media metrics. Your task is to generate a comprehensive, well-formatted report based on TikTok account analytics data. The user will provide a JSON object containing various metrics about their TikTok posts. You should analyze this data and create a professional report with the following components:          
                ## Report Structure    
                1. **Executive Summary**: Start with a concise summary of the overall account performance, highlighting 3-5 key metrics.                
                2. **Engagement Metrics**: Create a table showing the total and average engagement metrics (views, likes, comments, shares, downloads, collects).     
                3. **Content Analysis**: Analyze the different types of content (AI-generated, VR, ads, e-commerce, professional) and their distribution.
                4. **Top Performing Content**: Create a table showing the videos with the highest metrics in different categories.
                5. **Posting Frequency**: Analyze posting patterns including daily and weekly averages, and recent posting activity.
                6. **Strategic Insights**: Provide 3-5 data-backed insights and recommendations based on the metrics.
                
                ## Formatting Guidelines  
                - Use markdown tables for presenting numeric data
                - Include section headers with clear hierarchy
                - Use bullet points for listing insights and recommendations
                - Bold important numbers and key findings
                - Use emoji sparingly to enhance readability (💡 for insights, 📈 for growth metrics, etc.)
                - Format large numbers with commas for better readability
                - Round decimals to 2 places for averages and percentages
                
                ## Response Tone  
                - Professional but accessible
                - Data-driven with clear interpretations
                - Focus on actionable insights
                - Avoid overly technical jargon unless necessary
                
                When you receive the JSON data, parse it carefully and organize the information logically in your report. Pay special attention to highlighting notable patterns, outliers, and potential opportunities for improvement.     
                Your report should be comprehensive enough to provide value but concise enough to be quickly digestible. Aim for a report that would take 3-5 minutes to read thoroughly.
                """,
            "post_trend_analysis": """# System Prompt: Social Media Performance Analysis

            You are a data analyst specializing in social media analytics. Your task is to transform the provided JSON data into a readable format and conduct an insightful analysis.
    
            ## Instructions:
            
            1. Parse the JSON data containing post trends and interaction metrics.
            2. Create a well-formatted markdown table with the following columns (YOU MUST INCLUDE ALL DATES):
               - Date
               - Post Count
               - Digg Count (Likes)
               - Comment Count
               - Share Count
               - Play Count (Views)
            
            3. Calculate and highlight key performance metrics:
               - Days with highest post frequency
               - Days with highest engagement metrics (diggs, comments, shares, plays)
               - Weekly and monthly trends
               - Ratio of interactions to posts
               - Engagement rate calculation
            
            4. Produce a comprehensive report with the following sections:
               - Executive Summary (overall performance)
               - Content Performance (post frequency and timing analysis)
               - Audience Engagement (interaction metrics analysis)
               - Key Insights (highlighting notable patterns and anomalies)
               - Recommendations (based on data patterns)
            
            5. Format the report professionally with proper headings, bullet points, and emphasis on key findings.
            
            6. Include visual descriptions of trends that would be useful for the content creator.
            
            7. Present the data in a way that's accessible and actionable for the content creator.""",
            "post_duration_and_time": """# System Prompt: Video Content Distribution Visualization
            You are a data visualization specialist focusing on content creator analytics. Your task is to create and explain tables that visualize the distribution patterns in the provided data.
            ## Instructions:
            
            1. Parse the provided JSON data containing two key distribution metrics:
               - Video duration distribution (how long the videos are)
               - Publishing time distribution (when videos are published during the day)
            
            2. Create two clear and visually distinct table:
               - Table 1: Video Duration Distribution
               - Table 2: Publishing Time Distribution
            
            3. For each Table:
               - Use an appropriate color scheme that differentiates segments clearly
               - Include percentage labels on each segment
               - Add a clear title and legend
               - Ensure the segments are ordered logically (e.g., duration from shortest to longest)
            
            4. Provide a brief analysis of each chart, highlighting:
               - The most common video duration
               - The preferred publishing time
               - Any notable patterns or imbalances in the distribution
            
            5. Suggest actionable insights based on the data, such as:
               - Optimal video length based on current patterns
               - Best times to publish for increased engagement
               - Potential opportunities in underutilized duration ranges or time slots
            
            6. Format your response as a well-structured markdown document with:
               - Clear headings
               - Tables formatted for readability
               - Analysis text separated from code
               - A concise summary
            
            7. Ensure your visualization code is complete and ready to execute with the provided data.
            """,
            "post_hashtags":"""# System Prompt: Social Media Hashtag Analysis
                You are a social media analytics expert specializing in content categorization and trend analysis. Your task is to analyze a collection of hashtags from a content creator and provide structured insights.     
                ## Instructions:             
                1. Parse the provided hashtag data containing hashtag names, usage counts, and unique identifiers.            
                2. Create a well-formatted markdown table with the following columns:
                   - Hashtag Name (without the # symbol)
                   - Usage Count
                   - Hashtag ID           
                3. start the table by usage count in descending order to highlight the most frequently used hashtags.             
                4. Perform a comprehensive analysis of the hashtags, including:
                   - Industry/vertical identification (e.g., beauty, pharmacy, wellness, fashion)
                   - Product categories (e.g., skincare, cosmetics, pharmaceuticals)
                   - Content types or themes (e.g., tutorials, product reviews, tips)
                   - Target audience demographics
                   - Language analysis (identify primary language and any multilingual strategies)         
                5. Group related hashtags into logical categories based on their themes and purposes.
                
                6. Identify the top 5 most important hashtags and explain their significance to the creator's content strategy.
                
                7. Produce a detailed report with the following sections:
                   - Executive Summary
                   - Hashtag Usage Table
                   - Content Category Analysis
                   - Product/Service Focus
                   - Audience Targeting Strategy
                   - Language & Geographic Focus
                   - Key Hashtag Analysis
                   - Strategic Recommendations
                
                8. Format your analysis as a professional report using proper markdown with clear section headers, bullet points, and emphasis where appropriate.
                
                9. Provide actionable recommendations on:
                   - Hashtag optimization opportunities
                   - Underutilized hashtag categories
                   - Strategic hashtag combinations
                   - Potential new hashtags to explore             
                Your analysis should be thorough, data-driven, and provide valuable insights that the content creator can implement to improve their social media strategy.
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
            user_prompt = f"Generate a report for the {analysis_type} based on the following data:\n{json.dumps(data, ensure_ascii=False)}, for user {uniqueId}"

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

    async def fetch_user_posts_stats(self, url: str, max_post: Optional[int]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        分析用户/达人的发布作品统计
        """
        post_count = 0
        start_time = time.time()
        posts_raw_data = []
        posts_stats = {}
        total_posts = await self.user_collector.fetch_total_posts_count(url)
        max_post = min(max_post, total_posts)

        logger.info("正在分析发布作品统计...")
        try:
            # 采集用户发布的作品数据
            async for posts in self.user_collector.collect_user_posts(url):
                cleaned_posts = await self.user_cleaner.clean_user_posts(posts)
                if cleaned_posts:
                    if post_count+ len(cleaned_posts) <= max_post:
                        posts_raw_data.extend(cleaned_posts)
                        post_count += len(cleaned_posts)
                        yield{
                            'user_profile_url': url,
                            'is_complete': False,
                            'message': f'已采集{post_count}条作品数据, 进度: {post_count}/{max_post}...',
                            'total_posts': total_posts,
                            'posts_stats': posts_stats,
                            'posts_raw_data': posts_raw_data,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'processing_time': round(time.time() - start_time, 2)
                        }
                    else:
                        posts_raw_data.extend(cleaned_posts[:max_post - post_count])
                        post_count = max_post
                        logger.info(f"已采集{post_count}条作品数据, 完成")
                        break
            # 使用pandas进行数据处理
            df = pd.DataFrame(posts_raw_data)

            # 转换时间并按发布时间排序 - 使用unit='s'指定输入是秒级时间戳
            df["create_time"] = pd.to_datetime(df["create_time"], unit='s')
            # 按照最近发布时间排序
            df = df.sort_values("create_time")
            df["day"] = df["create_time"].dt.date

            stats = {
                "total_posts": int(df.shape[0]),
                "average_collects": float(df["collect_count"].mean()),
                "average_likes": float(df["digg_count"].mean()),
                "average_downloads": float(df["download_count"].mean()),
                "average_views": float(df["play_count"].mean()),
                "average_comments": float(df["comment_count"].mean()),
                "average_shares": float(df["share_count"].mean()),
                "average_whatsapp_shares": float(df["whatsapp_share_count"].mean()),
                "total_likes": int(df["digg_count"].sum()),
                "total_comments": int(df["comment_count"].sum()),
                "total_shares": int(df["share_count"].sum()),
                "total_whatsapp_shares": int(df["whatsapp_share_count"].sum()),
                "total_views": int(df["play_count"].sum()),
                "total_downloads": int(df["download_count"].sum()),
                "total_ai_videos": int(df["created_by_ai"].eq(True).sum()),
                "total_vr_videos": int(df["is_vr"].eq(True).sum()),
                "total_ads_videos": int(df["is_ads"].eq(True).sum()),
                "total_ec_videos": int(df["is_ec_video"].eq(1).sum()),
                "total_risk_videos": int((df["in_reviewing"] & df["is_prohibited"]).sum()),
                "total_recommendation_videos": int(df["is_nff_or_nr"].eq(False).sum()),
                "total_professional_generated_videos": int(df["is_pgcshow"].eq(True).sum()),
                "highest_likes": {
                    "count": int(df["digg_count"].max()),
                    "video": str(df.loc[df["digg_count"].idxmax()]["aweme_id"]),
                    "publish_date": str(df.loc[df["digg_count"].idxmax()]["create_time"])
                },
                "highest_comments": {
                    "count": int(df["comment_count"].max()),
                    "video": str(df.loc[df["comment_count"].idxmax()]["aweme_id"]),
                    "publish_date": str(df.loc[df["comment_count"].idxmax()]["create_time"])
                },
                "highest_shares": {
                    "count": int(df["share_count"].max()),
                    "video": str(df.loc[df["share_count"].idxmax()]["aweme_id"]),
                    "publish_date": str(df.loc[df["share_count"].idxmax()]["create_time"])
                },
                "highest_downloads": {
                    "count": int(df["download_count"].max()),
                    "video": str(df.loc[df["download_count"].idxmax()]["aweme_id"]),
                    "publish_date": str(df.loc[df["download_count"].idxmax()]["create_time"])
                },
                "highest_views": {
                    "count": int(df["play_count"].max()),
                    "video": str(df.loc[df["play_count"].idxmax()]["aweme_id"]),
                    "publish_date": str(df.loc[df["play_count"].idxmax()]["create_time"])
                },
                "highest_whatsapp_shares": int(df["whatsapp_share_count"].max()),
                "average_video_duration": float(round(df["duration"].mean() / 1000, 2)),
                "post_per_day": float(df["day"].value_counts().mean()),
                "post_per_week": float(df["day"].value_counts().mean() * 7),
                "latest_week_post_count": {str(k): int(v) for k, v in
                                           df["day"].value_counts().head(7).to_dict().items()}
            }

            report_url= await self.generate_analysis_report(url, 'post_stats_analysis', stats)


            logger.info(f"已完成用户 {url} 发布作品统计分析")

            yield {
                'user_profile_url': url,
                'is_complete': True,
                'message': f'已完成发布作品统计分析',
                'report_url': report_url,
                'total_posts': total_posts,
                'posts_stats': stats,
                'posts_raw_data': posts_raw_data,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }
        except Exception as e:
            logger.error(f"分析指定用户发布作品统计时发生错误: {str(e)}")
            yield {
                'user_profile_url': url,
                'is_complete': False,
                'error': str(e),
                'message': f"分析发布作品统计时发生错误: {str(e)}",
                'total_posts': total_posts,
                'posts_stats': posts_stats,
                'posts_raw_data': posts_raw_data,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }

    async def fetch_user_posts_trend(self, url: str, time_interval:str = '90D') -> AsyncGenerator[Dict[str, Any], None]:
        """
        分析用户/达人的发布作品趋势

        Args:
            url: 用户/达人主页URL
            time_interval: 时间间隔，默认30天

        Returns:
            Dict包含:
            - post_trend: 发布趋势数据
            - interaction_trend: 互动趋势数据
        """
        post_count = 0
        start_time = time.time()
        posts_raw_data = []
        total_posts = await self.user_collector.fetch_total_posts_count(url)

        logger.info("正在分析发布作品趋势统计...")

        try:
            # 采集用户发布的作品数据
            async for posts in self.user_collector.collect_user_posts(url):
                cleaned_posts = await self.user_cleaner.clean_user_posts(posts)
                if cleaned_posts:
                    post_count += len(cleaned_posts)
                    if post_count <= total_posts:
                        posts_raw_data.extend(cleaned_posts)
                        yield{
                            'user_profile_url': url,
                            'is_complete': False,
                            'message': f'已采集{post_count}条作品数据..., 进度: {post_count}/{total_posts}...',
                            'total_posts': total_posts,
                            #'posts_raw_data': posts_raw_data,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'processing_time': round(time.time() - start_time, 2)
                        }
                    elif post_count > total_posts:
                        posts_raw_data.extend(cleaned_posts[:total_posts - post_count])
                        post_count = total_posts
                        logger.info(f"已采集{post_count}条作品数据, 准备分析发布趋势")
                        yield {
                            'user_profile_url': url,
                            'is_complete': False,
                            'message': f'已采集{post_count}条作品数据, 准备分析发布趋势...',
                            'total_posts': total_posts,
                            #'posts_raw_data': posts_raw_data,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'processing_time': round(time.time() - start_time, 2)
                        }
                        break
                else:
                    logger.info(f"已采集{post_count}条作品数据, 准备分析发布趋势")
                    yield {
                        'user_profile_url': url,
                        'is_complete': False,
                        'message': f'已采集{post_count}条作品数据, 准备分析发布趋势...',
                        'total_posts': total_posts,
                        #'posts_raw_data': posts_raw_data,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'processing_time': round(time.time() - start_time, 2)
                    }
                    break

            # 使用pandas进行数据处理
            df = pd.DataFrame(posts_raw_data)

            # 转换时间并按发布时间排序 - 使用unit='s'指定输入是秒级时间戳
            df["create_time"] = pd.to_datetime(df["create_time"], unit='s')
            df = df.sort_values("create_time")

            # 计算时间范围 - 使用当前时间作为结束时间
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(time_interval)

            # 筛选时间范围内的数据
            df = df[df["create_time"].between(start_date, end_date)]

            # 生成日期序列作为基准
            date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')

            # 统计每日发布数量
            df["date"] = df["create_time"].dt.date
            daily_posts = df["date"].value_counts().reindex(date_range.date, fill_value=0)

            # 统计每日互动数据, 包括点赞数，评论数，分享数，播放数
            interaction_metrics = ["digg_count", "comment_count", "share_count", "play_count"]
            daily_interactions = df.groupby("date")[interaction_metrics].sum()
            daily_interactions = daily_interactions.reindex(date_range.date, fill_value=0)

            # 构建返回数据 - 确保按日期排序
            daily_posts = daily_posts.sort_index()
            daily_interactions = daily_interactions.sort_index()

            trends_data = {
                "post_trend": {
                    "x": [d.strftime("%Y-%m-%d") for d in daily_posts.index],
                    "y": daily_posts.values.tolist()
                },
                "interaction_trend": {
                    metric: {
                        "x": [d.strftime("%Y-%m-%d") for d in daily_interactions.index],
                        "y": daily_interactions[metric].values.tolist()
                    }
                    for metric in interaction_metrics
                }
            }
            # 将trend data 用json格式保存
            # print(json.dumps(trends_data, indent=4))
            uniqueId = url.split("@")[-1]

            report_url= await self.generate_analysis_report(uniqueId, 'post_trend_analysis', trends_data)

            yield{
                'user_profile_url': url,
                'is_complete': True,
                'message': f'已完成发布作品趋势分析',
                'report_url': report_url,
                'total_posts': total_posts,
                #'posts_raw_data': posts_raw_data,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }

        except Exception as e:
            logger.error(f"分析发布趋势时发生错误: {str(e)}")
            yield {
                'user_profile_url': url,
                'is_complete': False,
                'error': str(e),
                'message': f"分析发布趋势时发生错误: {str(e)}",
                'total_posts': total_posts,
                'posts_raw_data': posts_raw_data,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }
            return

    async def fetch_post_duration_and_time_distribution(self, url: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        分析用户/达人的发布作品时长分布
        """
        logger.info("正在分析发布作品的时长分布以及时间分布...")
        start_time = time.time()
        post_count = 0
        posts_raw_data = []
        duration_distribution = time_distribution = {}
        total_posts = await self.user_collector.fetch_total_posts_count(url)
        try:
            # 采集用户发布的作品数据
            async for posts in self.user_collector.collect_user_posts(url):
                cleaned_posts = await self.user_cleaner.clean_user_posts(posts)
                if cleaned_posts:
                    post_count += len(cleaned_posts)
                    if post_count <= total_posts:
                        posts_raw_data.extend(cleaned_posts)
                        yield {
                            'user_profile_url': url,
                            'is_complete': False,
                            'message': f'已采集{post_count}条作品数据..., 进度: {post_count}/{total_posts}...',
                            'total_posts': total_posts,
                            # 'posts_raw_data': posts_raw_data,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'processing_time': round(time.time() - start_time, 2)
                        }
                    elif post_count > total_posts:
                        posts_raw_data.extend(cleaned_posts[:total_posts - post_count])
                        post_count = total_posts
                        logger.info(f"已采集{post_count}条作品数据, 准备分析发布趋势")
                        yield {
                            'user_profile_url': url,
                            'is_complete': False,
                            'message': f'已采集{post_count}条作品数据, 准备分析发布趋势...',
                            'total_posts': total_posts,
                            # 'posts_raw_data': posts_raw_data,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'processing_time': round(time.time() - start_time, 2)
                        }
                        break
                else:
                    logger.info(f"已采集{post_count}条作品数据, 准备分析发布趋势")
                    yield {
                        'user_profile_url': url,
                        'is_complete': False,
                        'message': f'已采集{post_count}条作品数据, 准备分析发布趋势...',
                        'total_posts': total_posts,
                        # 'posts_raw_data': posts_raw_data,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'processing_time': round(time.time() - start_time, 2)
                    }
                    break

            # 使用pandas进行数据处理
            df = pd.DataFrame(posts_raw_data)

            # 将视频时长从毫秒转换为秒
            df["duration"] = df["duration"] / 1000

            # 根据视频时长分布统计， 0-15s, 15-30s, 30-60s, 60-120s, 120s以上
            bins = [0, 15, 30, 60, 120, float("inf")]
            labels = ["0-15s", "15-30s", "30-60s", "60-120s", "120s+"]
            df["duration_range"] = pd.cut(df["duration"], bins=bins, labels=labels)

            # 统计每个时长区间的视频数量
            duration_distribution = df["duration_range"].value_counts().to_dict()

            # 转换时间并按发布时间排序 - 使用unit='s'指定输入是秒级时间戳
            df["create_time"] = pd.to_datetime(df["create_time"], unit='s')
            df = df.sort_values("create_time")

            # 只根据小时提取时间，24小时制，0-5点为凌晨，6-11点为上午，12-17点为下午，18-23点为晚上
            df["hour"] = df["create_time"].dt.hour
            df["hour_range"] = pd.cut(df["hour"], bins=[0, 6, 12, 18, 24], labels=["Dawn/Early Morning", "Morning", "Afternoon", "Evening"])

            # 统计每个时间段的视频数量
            time_distribution = df["hour_range"].value_counts().to_dict()

            distributions = {
                "duration_distribution": duration_distribution,
                "time_distribution": time_distribution
            }
            print(json.dumps(distributions, indent=4))

            uniqueId = url.split("@")[-1]

            report_url = await self.generate_analysis_report(uniqueId, 'post_duration_and_time', distributions)

            yield {
                'user_profile_url': url,
                'is_complete': True,
                'message': f'已完成发布作品时长分布和时间分布分析',
                'report_url': report_url,
                'total_posts': total_posts,
                'duration_distribution': duration_distribution,
                'time_distribution': time_distribution,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }

        except Exception as e:
            logger.error(f"分析发布作品时长分布时发生错误: {str(e)}")
            yield {
                'user_profile_url': url,
                'is_complete': False,
                'error': str(e),
                'message': f"分析发布作品时长分布时发生错误: {str(e)}",
                'total_posts': total_posts,
                'duration_distribution': duration_distribution,
                'time_distribution': time_distribution,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }
            return

    async def fetch_post_hashtags(self, url: str, count: int) -> AsyncGenerator[Dict[str, Any], None]:
        """
        获取所有的话题，排名使用率最高的话题， 并且生成报告
        """
        logger.info("正在获取话题数据...")
        start_time = time.time()
        post_count = 0
        posts_raw_data = []
        hashtags = {}
        total_posts = await self.user_collector.fetch_total_posts_count(url)
        try:
            # 采集用户发布的作品数据
            async for posts in self.user_collector.collect_user_posts(url):
                cleaned_posts = await self.user_cleaner.clean_user_posts(posts)
                if cleaned_posts:
                    post_count += len(cleaned_posts)
                    if post_count <= total_posts:
                        posts_raw_data.extend(cleaned_posts)
                        yield {
                            'user_profile_url': url,
                            'is_complete': False,
                            'message': f'已采集{post_count}条作品数据..., 进度: {post_count}/{total_posts}...',
                            'total_posts': total_posts,
                            # 'posts_raw_data': posts_raw_data,
                            'top_hashtags': hashtags,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'processing_time': round(time.time() - start_time, 2)
                        }
                    elif post_count > total_posts:
                        posts_raw_data.extend(cleaned_posts[:total_posts - post_count])
                        post_count = total_posts
                        logger.info(f"已采集{post_count}条作品数据, 准备分析发布趋势")
                        yield {
                            'user_profile_url': url,
                            'is_complete': False,
                            'message': f'已采集{post_count}条作品数据, 准备分析发布趋势...',
                            'total_posts': total_posts,
                            # 'posts_raw_data': posts_raw_data,
                            'top_hashtags': hashtags,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'processing_time': round(time.time() - start_time, 2)
                        }
                        break
                else:
                    logger.info(f"已采集{post_count}条作品数据, 准备分析发布趋势")
                    yield {
                        'user_profile_url': url,
                        'is_complete': False,
                        'message': f'已采集{post_count}条作品数据, 准备分析发布趋势...',
                        'total_posts': total_posts,
                        'top_hashtags': hashtags,
                        # 'posts_raw_data': posts_raw_data,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'processing_time': round(time.time() - start_time, 2)
                    }
                    break

            # 使用pandas进行数据处理
            df = pd.DataFrame(posts_raw_data)

            # 获取所有的话题, 以及每个话题的使用次数
            all_hashtags = df["hashtags"]
            hashtags_regroup = {}
            for hashtags in all_hashtags:
                if hashtags is None:
                    continue
                hashtags = json.loads(hashtags)
                for name, id in hashtags.items():
                    if name in hashtags_regroup:
                        hashtags_regroup[name]["count"] += 1
                    else:
                        hashtags_regroup[name] = {"count": 1, "id": id}

            # 获取使用率最高的话题
            count = min(count, len(hashtags_regroup))
            hashtags = sorted(hashtags_regroup.items(), key=lambda x: x[1]["count"], reverse=True)[:count]
            hashtags_dict = {hashtag: data for hashtag, data in hashtags}

            uniqueID = url.split("@")[-1]

            report_url = await self.generate_analysis_report(uniqueID, 'post_hashtags', hashtags_dict)

            yield {
                'user_profile_url': url,
                'is_complete': True,
                'message': f'已完成获取话题数据',
                'report_url': report_url,
                'total_posts': total_posts,
                'top_hashtags': hashtags,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }

        except Exception as e:
            logger.error(f"❌ 获取话题数据时发生错误: {str(e)}")
            yield {
                'user_profile_url': url,
                'is_complete': False,
                'error': str(e),
                'message': f"获取话题数据时发生错误: {str(e)}",
                'total_posts': total_posts,
                'top_hashtags': hashtags,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }

    async def fetch_post_creator_analysis(self, url: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        综合分析创作者视频，包括热门视频、广告/带货视频、AI/VR视频、风险视频

        Args:
            url: 用户个人主页URL

        Yields:
            Dict: 各个阶段的分析结果，视频信息仅包含aweme_id, desc, download_addr, create_time
        """
        logger.info("🔍 开始全面分析创作者视频数据...")
        start_time = time.time()
        post_count = 0
        posts_raw_data = []
        analysis_results = {}

        try:
            # 获取用户总发布作品数
            total_posts = await self.user_collector.fetch_total_posts_count(url)

            # 采集用户发布的作品数据
            async for posts in self.user_collector.collect_user_posts(url):
                cleaned_posts = await self.user_cleaner.clean_user_posts(posts)
                if cleaned_posts:
                    post_count += len(cleaned_posts)
                    if post_count <= total_posts:
                        posts_raw_data.extend(cleaned_posts)
                        yield {
                            'user_profile_url': url,
                            'is_complete': False,
                            'message': f'已采集{post_count}条作品数据..., 进度: {post_count}/{total_posts}...',
                            'total_posts': total_posts,
                            'analysis_results': analysis_results,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'processing_time': round(time.time() - start_time, 2)
                        }
                    elif post_count > total_posts:
                        posts_raw_data.extend(cleaned_posts[:total_posts - post_count])
                        post_count = total_posts
                        logger.info(f"已采集{post_count}条作品数据, 准备开始分析...")
                        yield {
                            'user_profile_url': url,
                            'is_complete': False,
                            'message': f'已采集{post_count}条作品数据, 准备开始分析...',
                            'total_posts': total_posts,
                            'analysis_results': analysis_results,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'processing_time': round(time.time() - start_time, 2)
                        }
                        break
                else:
                    logger.info(f"已采集{post_count}条作品数据, 准备开始分析...")
                    yield {
                        'user_profile_url': url,
                        'is_complete': False,
                        'message': f'已采集{post_count}条作品数据, 准备开始分析...',
                        'total_posts': total_posts,
                        'analysis_results': analysis_results,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'processing_time': round(time.time() - start_time, 2)
                    }
                    break

            # 使用pandas进行数据处理
            df = pd.DataFrame(posts_raw_data)

            # 定义一个函数，用于简化视频数据，只保留指定字段
            def simplify_video_data(videos_list):
                simplified_videos = []
                for video in videos_list:
                    simplified_videos.append({
                        'aweme_id': video.get('aweme_id'),
                        'desc': video.get('desc'),
                        'download_addr': video.get('download_addr'),
                        'create_time': video.get('create_time')
                    })
                return simplified_videos

            # 1. 分析热门视频
            logger.info("📊 正在分析热门视频...")
            yield {
                'user_profile_url': url,
                'is_complete': False,
                'message': '正在分析热门视频...',
                'total_posts': total_posts,
                'analysis_results': analysis_results,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }

            # 获取热门视频，按照点赞数排序，取前5
            hot_videos_digg = df.sort_values("digg_count", ascending=False).head(5).to_dict(orient="records")

            # 获取热门视频，按照播放量数排序，取前5
            hot_videos_views = df.sort_values("play_count", ascending=False).head(5).to_dict(orient="records")

            # 获取热门视频，按照评论数排序，取前5
            hot_videos_comments = df.sort_values("comment_count", ascending=False).head(5).to_dict(orient="records")

            # 获取热门视频，按照分享数排序，取前5
            hot_videos_shares = df.sort_values("share_count", ascending=False).head(5).to_dict(orient="records")

            # 将它们合并，根据aweme_id去重
            hot_videos = []
            seen_ids = set()

            for video in hot_videos_digg + hot_videos_views + hot_videos_comments + hot_videos_shares:
                video_id = video['aweme_id']
                if video_id not in seen_ids:
                    seen_ids.add(video_id)
                    hot_videos.append(video)

            # 简化热门视频数据
            simplified_hot_videos = simplify_video_data(hot_videos)

            # 获取置顶视频
            top_videos = df[df["is_top"].eq(True)].to_dict(orient="records")

            # 简化置顶视频数据
            simplified_top_videos = simplify_video_data(top_videos)

            analysis_results["hot_videos"] = {
                "hot_videos": simplified_hot_videos,
                "top_videos": simplified_top_videos,
                "top_videos_count": len(simplified_top_videos)
            }

            # 2. 分析广告/带货视频
            logger.info("📊 正在分析广告/带货视频...")
            yield {
                'user_profile_url': url,
                'is_complete': False,
                'message': '正在分析广告/带货视频...',
                'total_posts': total_posts,
                'analysis_results': analysis_results,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }

            # 获取广告视频
            ads_videos = df[df["is_ads"].eq(True)].to_dict(orient="records")

            # 简化广告视频数据
            simplified_ads_videos = simplify_video_data(ads_videos)

            # 获取电商视频
            ec_videos = df[df["is_ec_video"].eq(True)].to_dict(orient="records")

            # 简化电商视频数据
            simplified_ec_videos = simplify_video_data(ec_videos)

            analysis_results["commerce_videos"] = {
                "ads_videos_count": len(simplified_ads_videos),
                'ec_videos_count': len(simplified_ec_videos),
                "ads_videos": simplified_ads_videos,
                'ec_videos': simplified_ec_videos
            }

            # 3. 分析AI/VR生成视频
            logger.info("📊 正在分析AI/VR生成视频...")
            yield {
                'user_profile_url': url,
                'is_complete': False,
                'message': '正在分析AI/VR生成视频...',
                'total_posts': total_posts,
                'analysis_results': analysis_results,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }

            # 获取AI生成视频
            ai_videos = df[df["created_by_ai"].eq(True)].to_dict(orient="records")

            # 简化AI生成视频数据
            simplified_ai_videos = simplify_video_data(ai_videos)

            # 获取VR视频
            vr_videos = df[df["is_vr"].eq(True)].to_dict(orient="records")

            # 简化VR视频数据
            simplified_vr_videos = simplify_video_data(vr_videos)

            analysis_results["synthetic_videos"] = {
                "ai_videos_count": len(simplified_ai_videos),
                'vr_videos_count': len(simplified_vr_videos),
                "ai_videos": simplified_ai_videos,
                'vr_videos': simplified_vr_videos
            }

            # 4. 分析风险视频
            logger.info("📊 正在分析风险视频...")
            yield {
                'user_profile_url': url,
                'is_complete': False,
                'message': '正在分析风险视频...',
                'total_posts': total_posts,
                'analysis_results': analysis_results,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }

            # 获取风险视频
            risk_videos = df[df["in_reviewing"] | df["is_prohibited"]].to_dict(orient="records")

            # 简化风险视频数据
            simplified_risk_videos = simplify_video_data(risk_videos)

            analysis_results["risk_videos"] = {
                "risk_videos_count": len(simplified_risk_videos),
                "risk_videos": simplified_risk_videos
            }

            uniqueID = url.split("@")[-1]
            report_url = await self.generate_analysis_report(uniqueID, 'post_creator_analysis', analysis_results)

            # 完成所有分析，返回最终结果
            yield {
                'user_profile_url': url,
                'is_complete': True,
                'message': '分析完成',
                'report_url': report_url,
                'total_posts': total_posts,
                'analysis_results': analysis_results,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }

        except Exception as e:
            logger.error(f"分析创作者视频时发生错误: {str(e)}")
            yield {
                'user_profile_url': url,
                'is_complete': False,
                'error': str(e),
                'message': f"分析创作者视频时发生错误: {str(e)}",
                'total_posts': total_posts,
                'analysis_results': analysis_results,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': round(time.time() - start_time, 2)
            }

        """
        获取用户/达人的粉丝画像
        """

        data = kwargs.get('data')
        logger.info("📊 正在分析用户粉丝画像...")
        with open(f"{config.DATA_DIR}/fans_analysis.json", "w") as f:
            json.dump(data, f)
        return data



async def main():
    crawler = UserCollector()
    cleaner = UserCleaner()
    analyzer = UserAgent()

    user_url = "https://www.tiktok.com/@galileofarma"

    # 测试fetch_user_posts_trend
    async for data in analyzer.fetch_post_hashtags(user_url, 30):
        print(data)


if __name__ == "__main__":
    asyncio.run(main())

