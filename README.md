# TikTok 评论分析 API

这个项目是一个基于 FastAPI 的 Web 服务，用于获取、分析 TikTok 视频评论数据，并识别潜在客户。

## 功能特点

- **评论获取**：从 TikTok 视频中获取评论数据
- **购买意图分析**：分析评论中的购买意图和用户情感
- **潜在客户识别**：根据评论内容识别潜在客户，并计算参与度分数
- **异步处理**：全面支持异步操作，提高性能
- **批处理**：支持大量评论的批处理分析
- **错误处理**：完善的错误处理和日志记录
- **Swagger UI**：自动生成的 API 文档和测试界面

## 安装

### 前提条件

- Python 3.8+
- pip

### 步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/tiktok-comment-api.git
cd tiktok-comment-api
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 复制环境变量模板并填写必要的 API 密钥：

```bash
cp .env.example .env
```

然后编辑 `.env` 文件，填写以下信息：
- `OPENAI_API_KEY`: OpenAI API 密钥（用于 ChatGPT）
- `ANTHROPIC_API_KEY`: Anthropic API 密钥（用于 Claude，可选）
- `TIKHUB_API_KEY`: TikHub API 密钥（用于获取 TikTok 评论）
- `TIKHUB_BASE_URL`: TikHub API 基础 URL

## 运行

启动开发服务器：

```bash
python -m app.main
```

或者使用 uvicorn 直接运行：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

服务器将在 http://localhost:8000 运行。API 文档可在 http://localhost:8000/docs 访问。

## API 端点

### 评论获取

- `POST /api/v1/comments/fetch`: 获取视频评论
- `GET /api/v1/comments/{aweme_id}`: 使用 GET 方法根据 ID 获取视频评论

### 评论分析

- `POST /api/v1/comments/analysis/purchase-intent`: 分析购买意图
- `POST /api/v1/comments/analysis/potential-customers`: 识别潜在客户

### 工具

- `GET /api/v1/comments/health`: 健康检查

## 配置

主要配置选项可以通过环境变量或 `.env` 文件设置：

| 参数 | 描述 | 默认值 |
|------|-------------|---------|
| `HOST` | 服务器主机 | 0.0.0.0 |
| `PORT` | 服务器端口 | 8000 |
| `DEBUG` | 调试模式 | false |
| `LOG_LEVEL` | 日志级别 | info |
| `DEFAULT_BATCH_SIZE` | 默认批处理大小 | 30 |
| `MAX_BATCH_SIZE` | 最大批处理大小 | 100 |
| `DEFAULT_AI_MODEL` | 默认 AI 模型 | gpt-4o-mini |

## 开发

### 项目结构

```
tiktok-comment-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用主入口
│   ├── config.py            # 配置文件
│   ├── dependencies.py      # 依赖注入
│   ├── api/
│   │   ├── routes/
│   │   │   ├── comments.py  # 评论相关 API 路由
│   │   ├── models/
│   │   │   ├── comments.py  # 评论相关数据模型
│   │   │   ├── responses.py # 通用响应模型
│   ├── core/
│   │   ├── comment_agent.py # 评论代理类
│   │   ├── exceptions.py    # 自定义异常类
│   ├── services/
│   │   ├── data_etl/
│   │   │   ├── crawler/
│   │   │   │   ├── comment_crawler.py  # 爬虫类
│   │   │   ├── cleaner/
│   │   │   │   ├── comment_cleaner.py  # 数据清洗类
│   │   ├── ai_models/
│   │   │   ├── chatgpt.py   # ChatGPT 客户端
│   │   │   ├── claude.py    # Claude 客户端
│   ├── utils/
│   │   ├── logger.py        # 日志工具
├── agents/
│   ├── __init__.py
│   ├── customer_agent.py # 评论代理类
```

### 添加新功能

1. 在 `app/api/models/` 中定义新的数据模型
2. 在 `app/core/comment_agent.py` 中实现功能逻辑
3. 在 `app/api/routes/comments.py` 中添加新的端点

## 错误处理

API 使用标准的 HTTP 状态码，并返回详细的错误信息：

- `400 Bad Request`: 输入参数验证失败
- `401 Unauthorized`: API 密钥缺失或无效
- `404 Not Found`: 请求的资源不存在
- `429 Too Many Requests`: 超过速率限制
- `500 Internal Server Error`: 服务器内部错误
- `502 Bad Gateway`: 外部 API 调用失败

## 参与贡献

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

## 许可证

[MIT](LICENSE)