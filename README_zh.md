<div align="center">
<h1><a href="https://github.com/Agentfy-io/Agentfy">Agentify Sub-Agents API</a></h1>
<a href="https://github.com/Agentfy-io/Agentfy_API/README.md">English</a> | <a href="https://github.com/Agentfy-io/Agentfy_API/README_zh.md">简体中文</a>
</div>

这是 [Agentify](https://github.com/Agentfy-io/Agentfy) 的 API 组件，一个基于 FastAPI 的服务，提供对专业 AI 代理的访问。每个代理都通过自己的 API 端点暴露，实现模块化和专注的功能。

[![Python](https://img.shields.io/badge/python-3.11+-yellow)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/Agentfy-io/Agentfy_API.svg?style=social&label=Stars)](https://github.com/Agentfy-io/Agentfy_API)
[![GitHub forks](https://img.shields.io/github/forks/Agentfy-io/Agentfy_API.svg?style=social&label=Forks)](https://github.com/Agentfy-io/Agentfy_API)
[![GitHub issues](https://img.shields.io/github/issues/Agentfy-io/Agentfy_API.svg)](https://github.com/Agentfy-io/Agentfy_API/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Agentfy-io/Agentfy_API/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Agentfy-io/Agentfy_API/blob/main/LICENSE)
[![Made with ❤️](https://img.shields.io/badge/made%20with-%E2%9D%A4%EF%B8%8F-red)](https://github.com/callmeiks)

## 主要功能

### TikTok

#### 📝 评论分析:
- **购买意向**: 识别潜在买家。
- **情感分析**: 分析受众情感。
- **创作者-粉丝关系**: 分类用户（忠实粉丝、喷子等）。
- **毒性检测**: 检测恶意和负面评论。

#### 🕵️‍♂️ 网红/用户分析:
- **用户信息**: 提取网红/商业账户数据，趋势分析。
- **帖子数据**: 分析帖子趋势、视频长度、热门视频、话题标签。
- **风险视频**: 基于 TikTok 字段分析内容风险。

#### 📹 视频分析:
- **指标**: 观看次数、点赞、分享、视频长度, 商业价值。
- **下载与转录**: 下载视频并转录内容。
- **帧分析**: 使用 OpenCV 提取和分析帧。
- **OCR**: 提取视频中的文本（产品名称、价格等）。

### 生成器

#### 🎥 短视频脚本生成:
- **脚本生成**: 为短视频生成脚本。
- **音频生成**: 使用您自己的声音或任何声音为短视频生成音频。
- **小红书帖子生成**: 基于抖音视频生成小红书帖子。（需要 Claude API 密钥）


## 快速开始

### 先决条件

- Python 3.11+
- pip

### 安装

1. 克隆仓库:
```bash
git clone https://github.com/Agentfy-io/Agentfy_API
cd Agentfy_API
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

4. 用您的 API 密钥编辑 `.env` 文件:
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
TIKHUB_API_KEY=your_tikhub_key
TIKHUB_BASE_URL=your_tikhub_base_url
```

5. 您必须从以下服务获取自己的 API 密钥:
- **TikHub API 密钥**: [https://www.user.tikhub.io](https://www.user.tikhub.io) (必需)
- **OpenAI API 密钥**: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) (必需)
- **LemonFox API 密钥**: [https://lemonfox.ai](https://lemonfox.ai) (可选)
- **ElevenLabs API 密钥**: [https://beta.elevenlabs.io](https://beta.elevenlabs.io) (可选)

6. 运行服务器:
```bash
python -m .main
```
或
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

在 `http://localhost:8000` 访问 API，在 `http://localhost:8000/docs` 访问文档

7. 在执行 API 中的某个端点后，您将获得一个任务 ID，请将其放入相应的任务检查端点以检查任务的状态和结果。

## 🙏 赞助与支持
本项目由 [TikHub](https://tikhub.io) 赞助，这是一个赋予开发者和企业无缝 API 的平台，将社交媒体数据转化为可行的洞察。
他们支持对 TikTok、抖音、Instagram、YouTube、X (Twitter)、小红书、Lemon8、哔哩哔哩等的数据访问。

- **🏠 首页**: [https://www.tikhub.io](https://www.tikhub.io)
- **👨‍💻 Github**: [https://github.com/TikHub](https://github.com/TikHub)
- **⚡ 文档 (Swagger UI)**: [https://api.tikhub.io](https://api.tikhub.io)
- **🦊 文档 (Apifox UI)**: [https://docs.tikhub.io](https://docs.tikhub.io)
- **🍱 SDK**: [https://github.com/TikHub/TikHub-API-Python-SDK](https://github.com/TikHub/TikHub-API-Python-SDK)
- **🐙 演示代码 (GitHub)**: [https://github.com/TikHub/TikHub-API-Demo](https://github.com/TikHub/TikHub-API-Demo)
- **📶 API 状态**: [https://monitor.tikhub.io](https://monitor.tikhub.io)
- **📧 支持**: [Discord 服务器](https://discord.gg/aMEAS8Xsvz)


## 📬 联系方式

有问题、想贡献或需要帮助将 Agentfy 集成到您的技术栈中？

请随时联系：

- 📧 **电子邮件:** [lqiu314@gmail.com](mailto:lqiu314@gmail.com) 或 [evil0ctal1985@gmail.com](mailto:evil0ctal1985@gmail.com) 
- 🧑‍💻 **GitHub:** [@callmeiks](https://github.com/callmeiks) 或 [@Evil0ctal](https://github.com/Evil0ctal)
- 💡 让我们一起构建下一代**由代理驱动的数字基础设施**。