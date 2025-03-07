# Agentify API

Agentify 是一个基于 FastAPI 构建的强大社交媒体智能助手，专为用户、内容创作者和企业设计，帮助他们获取、分析和利用各类社交媒体平台的数据。

## 核心功能

- **数据采集**：从 TikTok 等社交平台获取评论、互动指标和其他数据
- **智能分析**：分析用户情感、购买意向和受众参与度
- **内容生成**：创建短视频、帖子、文案和音频内容
- **创作者监控**：跨平台追踪内容表现
- **多代理系统**：专业代理组件用于客户互动、情感分析、视频和音频生成

## 技术架构

Agentify 使用多代理架构，包含专业化组件：

- **客户代理**：识别潜在客户并分析用户参与度
- **情感代理**：分析内容的情感基调和用户响应
- **视频代理**：生成和处理视频内容
- **音频生成器**：创建音频内容和处理声音

## 安装

### 前提条件

- Python 3.8+
- pip

### 设置步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/agentify.git
cd agentify
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 复制环境变量模板并添加您的 API 密钥：

```bash
cp .env.example .env
```

4. 编辑 `.env` 文件，填写以下信息：

```
OPENAI_API_KEY=OpenAI_API_密钥（用于 GPT 模型）
ANTHROPIC_API_KEY=Anthropic_API_密钥（用于 Claude）
TIKHUB_API_KEY=TikHub_API_密钥（用于获取 TikTok 数据）
TIKHUB_BASE_URL=TikHub_API_基础_URL
```

5. 运行服务

启动开发服务器：

```bash
python -m app.main
```

或使用 uvicorn 直接运行：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API 将在 `http://localhost:8000` 上可用，交互式文档可在 `http://localhost:8000/docs` 访问。

## API 端点

### 社交媒体数据

```http
POST /api/v1/comments/fetch    # 获取视频评论
GET  /api/v1/comments/{aweme_id} # 获取特定视频 ID 的评论
```

### 数据分析

```http
POST /api/v1/comments/analysis/purchase-intent    # 分析评论中的购买意向
POST /api/v1/comments/analysis/potential-customers # 识别潜在客户
POST /api/v1/comments/analysis/sentiment          # 分析评论中的情感
```

### 内容生成

```http
POST /api/v1/content/video  # 生成视频内容
POST /api/v1/content/audio  # 生成音频内容
POST /api/v1/content/copy   # 为帖子生成文案
```

### 创作者工具

```http
GET /api/v1/monitor/performance # 监控内容表现
GET /api/v1/monitor/trends      # 追踪与您领域相关的热门话题
```

## 配置

主要配置选项可通过环境变量或 `.env` 文件设置：

| 参数                | 描述                  | 默认值          |
|---------------------|---------------------|--------------|
| HOST               | 服务器主机            | 0.0.0.0      |
| PORT               | 服务器端口            | 8000         |
| DEBUG              | 调试模式              | false        |
| LOG_LEVEL          | 日志级别              | info         |
| DEFAULT_BATCH_SIZE | 默认批处理大小        | 30           |
| MAX_BATCH_SIZE     | 最大批处理大小        | 100          |
| DEFAULT_AI_MODEL   | 默认 AI 模型          | gpt-4o-mini  |

## 项目结构

```plaintext
agentify/
├── __init__.py
├── agents/     # 代理组件
    ├── audio_generator.py     # 音频内容生成
    ├── customer_agent.py      # 客户分析和参与度
    ├── sentiment_agent.py     # 情感分析引擎
    ├── video_agent.py         # 视频全方位分析
├── app/
    ├── __init__.py
    ├── main.py            # FastAPI 应用入口
    ├── config.py          # 配置文件
    ├── dependencies.py    # 依赖注入
    ├── api/
    │   ├── routes/
    │   │   ├── audio.py      # 音频内容生成 API 路由
    │   │   ├── auth.py       # 用户认证
    │   │   ├── customer.py   # 客户分析 API 路由
    │   │   ├── sentiment.py  # 情感分析 API 路由
    │   │   ├── video.py      # 视频分析 API 路由
    │   ├── models/
    │   │   ├── comments.py  # 评论相关数据模型
    │   │   ├── responses.py # 通用响应模型
    ├── core/
    │   ├── exceptions.py    # 自定义异常
    ├── utils/
        ├── logger.py        # 日志工具
├── services/
    │   ├── crawler/    # 爬虫类
    │   │   ├── comment_crawler.py  
    │   ├── cleaner/    # 数据清洗
    │   │   ├── comment_cleaner.py  
    │   ├── ai_models/    # AI 模型服务
    │   │   ├── chatgpt.py   # ChatGPT 客户端
    │   │   ├── claude.py    # Claude 客户端
    │   │   ├── genny.py    # Genny 客户端
    │   │   ├── opencv.py    # OpenCV 视频处理 
```

## 开发指南

### 添加新功能

1. 在 `app/api/models/` 中定义新的数据模型。
2. 在 `agents/`代理文件中实现功能逻辑。
3. 在 `app/api/routes/` 中添加新的端点。

### 代理组件

- **客户代理**：处理用户识别、参与度评分和潜在客户生成
- **情感代理**：处理评论和反应中的情感内容
- **视频代理**：处理视频生成、编辑和优化
- **音频生成器**：创建配音、音乐和音效

## 错误处理

API 使用标准 HTTP 状态码，并返回详细的错误信息：

- **400 Bad Request**：输入验证失败
- **401 Unauthorized**：API 密钥缺失或无效
- **404 Not Found**：请求的资源不存在
- **429 Too Many Requests**：超过速率限制
- **500 Internal Server Error**：服务器内部错误
- **502 Bad Gateway**：外部 API 调用失败

## 参与贡献

```bash
Fork 仓库
创建功能分支：git checkout -b feature/amazing-feature
提交更改：git commit -m 'Add amazing feature'
推送到分支：git push origin feature/amazing-feature
提交 Pull Request
```

## 许可证

MIT