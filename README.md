<div align="center">
<h1><a href="https://github.com/Agentfy-io/Agentfy">Agentify Sub-Agents API</a></h1>
<a href="https://github.com/Agentfy-io/Agentfy_API/blob/master/README.md">English</a> | <a href="https://github.com/Agentfy-io/Agentfy_API/blob/master/README_zh.md">简体中文</a>
</div>

This is the API component of [Agentify](https://github.com/Agentfy-io/Agentfy), a FastAPI-based service that provides access to specialized AI agents. Each agent is exposed through its own API endpoints, enabling modular and focused functionality.

[![Python](https://img.shields.io/badge/python-3.11+-yellow)](https://www.python.org/)
[![GitHub issues](https://img.shields.io/github/issues/Agentfy-io/Agentfy_API.svg)](https://github.com/Agentfy-io/Agentfy_API/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Agentfy-io/Agentfy_API/pulls)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Agentfy-io/Agentfy_API/blob/main/LICENSE)
[![Made with ❤️](https://img.shields.io/badge/made%20with-%E2%9D%A4%EF%B8%8F-red)](https://github.com/Agentfy-io)

## Key Features

### TikTok

#### 📝 Comment Analysis:
- **Purchase Intent**: Identify potential buyers.
- **Sentiment**: Analyze audience sentiment.
- **Creator-Follower Relationship**: Classify users (loyal fans, trolls, etc.).
- **Toxicity**: Detect malicious and negative comments.

#### 🕵️‍♂️ Influencer/User Analysis:
- **User Info**: Extract influencer/business account data.
- **Post Data**: Analyze post trends, video length, top videos, hashtags.
- **Risk Video**: Analyze content risk based on TikTok fields.

#### 📹 Video Analysis:
- **Metrics**: Views, likes, shares, video length.
- **Download & Transcription**: Download videos and transcribe content.
- **Frame Analysis**: Extract frames and analyze using OpenCV.
- **OCR**: Extract text in videos (product names, prices, etc.).

### Generators

#### 🎥 Short Video Script Generation:
- **Script Generation**: Generate scripts for short videos.
- **Audio Generation**: Generate audio for short videos using your own voice or any voice.
- **Xiaohongshu Post Generation**: Generate Xiaohongshu posts based on Douyin videos. （Claude API key required）


## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Agentfy-io/Agentfy_API
cd Agentfy_API
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Edit `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
TIKHUB_API_KEY=your_tikhub_key
TIKHUB_BASE_URL=your_tikhub_base_url
```

5. You must get your own API keys from the following services:
- **TikHub API Key**: [https://www.user.tikhub.io](https://www.user.tikhub.io) (Required)
- **OpenAI API Key**: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)(Required)
- **LemonFox API Key**: [https://lemonfox.ai](https://lemonfox.ai) (Optional)
- **ElevenLabs API Key**: [https://beta.elevenlabs.io](https://beta.elevenlabs.io) (Optional)

6. Run the server:
```bash
python -m .main
```
or
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access the API at `http://localhost:8000` and documentation at `http://localhost:8000/docs`

7. After you execute one of the endpoints in the API, you will get a task id, please put it in the corresponding tasks checking endpoint to check the status and result of the task.

## 🙏 Sponsorship & Support
This project is sponsor by [TikHub](https://tikhub.io), a platform that empower developers and businesses with seamless APIs to transform social media data into actionable insights.
They support Data Access to TikTok, Douyin, Instagram, YouTube, X (Twitter), Xiaohongshu, Lemon8, Bilibili, and more.

- **🏠 Home**: [https://www.tikhub.io](https://www.tikhub.io)
- **👨‍💻 Github**: [https://github.com/TikHub](https://github.com/TikHub)
- **⚡ Documents (Swagger UI)**: [https://api.tikhub.io](https://api.tikhub.io)
- **🦊 Documents (Apifox UI)**: [https://docs.tikhub.io](https://docs.tikhub.io)
- **🍱 SDK**: [https://github.com/TikHub/TikHub-API-Python-SDK](https://github.com/TikHub/TikHub-API-Python-SDK)
- **🐙 Demo Code (GitHub)**: [https://github.com/TikHub/TikHub-API-Demo](https://github.com/TikHub/TikHub-API-Demo)
- **📶 API Status**: [https://monitor.tikhub.io](https://monitor.tikhub.io)
- **📧 Support**: [Discord Server](https://discord.gg/aMEAS8Xsvz)


## 📬 Contact

Have questions, want to contribute, or need help integrating Agentfy into your stack?

Feel free to reach out:

- 📧 **Email:** [lqiu314@gmail.com](mailto:lqiu314@gmail.com) OR [evil0ctal1985@gmail.com](mailto:evil0ctal1985@gmail.com) 
- 🧑‍💻 **GitHub:** [@callmeiks](https://github.com/callmeiks) OR [@Evil0ctal](https://github.com/Evil0ctal)
- 💡 Let's build the next generation of **agent-powered digital infrastructure** — together.
