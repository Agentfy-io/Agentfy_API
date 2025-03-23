from services.ai_models.chatgpt import ChatGPT
from services.ai_models.claude import Claude
from typing import Dict, Any, List, Optional, Union, Tuple
from app.utils.logger import setup_logger
from dotenv import load_dotenv
import shutil
import os
from app.config import settings
from fastapi import HTTPException
import asyncio
import uuid
import aiofiles
from datetime import datetime, timedelta
import aiohttp
from services.crawler.comment_crawler import VideoCollector, VideoCleaner
from services.ai_models.whisper import WhisperLemonFox
import tempfile
import time
import threading
from pathlib import Path
from enum import Enum
import json
import re
from services.ai_models.videoOCR import VideoOCR



# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量
load_dotenv()

class VideoSubtitlesAgent:
    def __init__(self, tikhub_api_key: Optional[str] = None, tikhub_base_url: Optional[str] = None):
        """

        Args:
            tikhub_api_key: TikHub API密钥
            tikhub_base_url: TikHub API基础URL
        """
        # 保存TikHub API配置
        self.tikhub_api_key = tikhub_api_key
        self.tikhub_base_url = tikhub_base_url

        # 初始化AI模型客户端
        self.chatgpt = ChatGPT()
        self.claude = Claude()

        self.headers = {
            'Authorization': f'Bearer {self.tikhub_api_key}',
            'Content-Type': 'application/json'
        }
        self.job_manager=JobManager()



        # 如果没有提供TikHub API密钥，记录警告
        if not self.tikhub_api_key:
            logger.warning("未提供TikHub API密钥，某些功能可能不可用")

    # 工具函数
    async def save_upload_file(self,upload_file) -> str:
        """保存上传的文件并返回文件路径"""
        file_id = str(uuid.uuid4())
        _, ext = os.path.splitext(upload_file.filename)

        if ext.lower() not in settings.SUPPORTED_VIDEO_FORMATS:
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

        file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{ext}")

        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await upload_file.read()
            if len(content) > settings.MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"文件大小超过限制: {settings.MAX_FILE_SIZE / (1024 * 1024)}MB")
            await out_file.write(content)

        return file_path

    async def process_video(self,
            file_path: Optional[str],
            aweme_id: Optional[str],
            source_language: str,
            target_language: str,
            subtitle_format: str,
            job_id: str
    ) -> Dict[str, Any]:
        """处理视频并生成字幕"""


        try:
            # 参数验证
            if not file_path and not aweme_id:
                raise HTTPException(status_code=400, detail="必须提供视频文件或TikTok视频ID")
            # 更新作业状态
            self.job_manager.update_job(job_id, JobStatus.PROCESSING, "正在处理视频...")

            # 获取视频
            video_path = None
            if file_path:
                video_path = await self.save_upload_file(file_path)
                logger.info(f"获取视频 {video_path} 的路径")
            elif aweme_id:
                self.job_manager.update_job(job_id, JobStatus.PROCESSING, "正在下载TikTok视频...")
                video_path = await self.download_tiktok_video(aweme_id, settings.UPLOAD_DIR)
            else:
                self.job_manager.update_job(job_id, JobStatus.FAILED, "没有提供视频来源")
                return {"error": "没有提供视频来源"}

            # 进行语音识别
            self.job_manager.update_job(job_id, JobStatus.PROCESSING, "正在进行语音识别...")
            transcript = await self.recognize_speech(video_path, source_language)

            # 如果需要翻译
            if source_language != target_language and source_language != "auto":
                self.job_manager.update_job(job_id, JobStatus.PROCESSING, "正在翻译字幕...")
                translated_text = await self.translate_text(transcript, source_language, target_language)
            else:
                translated_text = transcript

            # 生成字幕
            self.job_manager.update_job(job_id, JobStatus.PROCESSING, "正在生成字幕...")
            output_video_path, srt_path = await generate_subtitles(
                video_path=video_path,
                transcript=translated_text,
                output_dir=settings.OUTPUT_DIR,
                subtitle_format=subtitle_format
            )
            print(output_video_path, srt_path)

            # 将输出文件移动到静态目录
            output_filename = os.path.basename(output_video_path)
            srt_filename = os.path.basename(srt_path)

            static_video_path = os.path.join(settings.STATIC_DIR, output_filename)
            static_srt_path = os.path.join(settings.STATIC_DIR, srt_filename)

            shutil.move(output_video_path, static_video_path)
            shutil.move(srt_path, static_srt_path)

            # 生成URL
            base_url = "/static"
            video_url = f"{base_url}/{output_filename}"
            srt_url = f"{base_url}/{srt_filename}"

            # 更新作业状态为完成
            result = {
                "video_url": video_url,
                "srt_url": srt_url,
                "source_language": source_language,
                "target_language": target_language,
                "file_paths": {
                    "video": static_video_path,
                    "srt": static_srt_path
                }
            }

            self.job_manager.update_job(job_id, JobStatus.COMPLETED, "处理完成", result=result)
            logger.info(f"生成结果: {result}")
            return result

        except Exception as e:
            logger.error(f"处理视频出错: {str(e)}", exc_info=True)
            self.job_manager.update_job(job_id, JobStatus.FAILED, f"处理失败: {str(e)}")
            return {"error": str(e)}



    async def process_batch_videos(self,videos_data: List[Dict[str, Any]], source_language: str,
                                   target_language: str, subtitle_format: str, job_id: str):
        """批量处理视频"""
        if not videos_data:
            raise HTTPException(status_code=400, detail="必须提供至少一个视频源")

        try:

            self.job_manager.update_job(job_id, JobStatus.PROCESSING, "开始批量处理视频...")

            results = []
            total_videos = len(videos_data)
            processed_count = 0

            # 创建子作业ID
            sub_job_ids = []
            for _ in range(total_videos):
                sub_job_id = str(uuid.uuid4())
                sub_job_ids.append(sub_job_id)
                self.job_manager.create_job(sub_job_id, parent_job_id=job_id)

            # 并行处理视频
            tasks = []
            for i, video_data in enumerate(videos_data):
                file_path = FileWrapper(video_data.get("file_path"))
                aweme_id = video_data.get("aweme_id")

                task = asyncio.create_task(
                    self.process_video(
                        file_path=file_path,
                        aweme_id=aweme_id,
                        source_language=source_language,
                        target_language=target_language,
                        subtitle_format=subtitle_format,
                        job_id=sub_job_ids[i]
                    )
                )
                tasks.append(task)

            # 等待所有任务完成
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for i, result in enumerate(completed_results):
                processed_count += 1
                progress = int((processed_count / total_videos) * 100)

                if isinstance(result, Exception):
                    results.append({
                        "index": i,
                        "status": "failed",
                        "error": str(result)
                    })
                elif isinstance(result, dict) and "error" in result:
                    results.append({
                        "index": i,
                        "status": "failed",
                        "error": result["error"]
                    })
                else:
                    results.append({
                        "index": i,
                        "status": "completed",
                        "result": result
                    })

                # 更新批处理作业进度
                self.job_manager.update_job(
                    job_id,
                    JobStatus.PROCESSING,
                    f"已处理 {processed_count}/{total_videos} 个视频 ({progress}%)"
                )

            # 批处理完成
            self.job_manager.update_job(
                job_id,
                JobStatus.COMPLETED,
                f"批处理完成: {processed_count}/{total_videos} 个视频已处理",
                result={"results": results}
            )

            return {"results": results}

        except Exception as e:
            logger.error(f"批量处理视频出错: {str(e)}", exc_info=True)
            self.job_manager.update_job(job_id, JobStatus.FAILED, f"批处理失败: {str(e)}")
            return {"error": str(e)}

    async def extract_video_subtitles(self, file_path: Optional[str], aweme_id: Optional[str], job_id: str):
        """从视频中提取字幕"""

        if not file_path and not aweme_id:
            raise HTTPException(status_code=400, detail="必须提供视频文件或TikTok视频ID")

        try:

            self.job_manager.update_job(job_id, JobStatus.PROCESSING, "正在提取字幕...")

            # 获取视频
            video_path = None
            if file_path:
                video_path = await self.save_upload_file(file_path)
            elif aweme_id:
                self.job_manager.update_job(job_id, JobStatus.PROCESSING, "正在下载TikTok视频...")
                video_path = await self.download_tiktok_video(aweme_id, settings.UPLOAD_DIR)
            else:
                self.job_manager.update_job(job_id, JobStatus.FAILED, "没有提供视频来源")
                return {"error": "没有提供视频来源"}

            # 提取字幕
            self.job_manager.update_job(job_id, JobStatus.PROCESSING, "正在从视频中提取字幕...")
            subtitle_text, srt_path = await extract_subtitles(video_path, settings.OUTPUT_DIR)

            # 将SRT文件移动到静态目录
            srt_filename = os.path.basename(srt_path)
            static_srt_path = os.path.join(settings.STATIC_DIR, srt_filename)
            shutil.move(srt_path, static_srt_path)

            # 生成URL
            srt_url = f"/static/{srt_filename}"

            result = {
                "subtitle_text": subtitle_text,
                "srt_url": srt_url,
                "file_path": static_srt_path
            }

            self.job_manager.update_job(job_id, JobStatus.COMPLETED, "字幕提取完成", result=result)
            return result

        except Exception as e:
            logger.error(f"提取字幕出错: {str(e)}", exc_info=True)
            self.job_manager.update_job(job_id, JobStatus.FAILED, f"提取字幕失败: {str(e)}")
            return {"error": str(e)}

    async def remove_video_subtitles(self, file_path: Optional[str], aweme_id: Optional[str], job_id: str):
        """从视频中移除硬编码字幕"""
        if not file_path and not aweme_id:
            raise HTTPException(status_code=400, detail="必须提供视频文件或TikTok视频ID")

        try:
            self.job_manager.update_job(job_id, JobStatus.PROCESSING, "正在处理视频...")

            # 获取视频
            video_path = None
            if file_path:
                video_path = await self.save_upload_file(file_path)
            elif aweme_id:
                self.job_manager.update_job(job_id, JobStatus.PROCESSING, "正在下载TikTok视频...")
                video_path = await self.download_tiktok_video(aweme_id, settings.UPLOAD_DIR)
            else:
                self.job_manager.update_job(job_id, JobStatus.FAILED, "没有提供视频来源")
                return {"error": "没有提供视频来源"}

            # 移除字幕
            self.job_manager.update_job(job_id, JobStatus.PROCESSING, "正在移除硬编码字幕...")
            output_video_path = await remove_subtitles(video_path, settings.OUTPUT_DIR)

            # 将输出文件移动到静态目录
            output_filename = os.path.basename(output_video_path)
            static_video_path = os.path.join(settings.STATIC_DIR, output_filename)
            shutil.move(output_video_path, static_video_path)

            # 生成URL
            video_url = f"/static/{output_filename}"

            result = {
                "video_url": video_url,
                "file_path": static_video_path
            }

            self.job_manager.update_job(job_id, JobStatus.COMPLETED, "字幕移除完成", result=result)
            return result

        except Exception as e:
            logger.error(f"移除字幕出错: {str(e)}", exc_info=True)
            self.job_manager.update_job(job_id, JobStatus.FAILED, f"移除字幕失败: {str(e)}")
            return {"error": str(e)}




    async def download_file(self,url: str, output_path: str) -> None:
        """
        异步下载文件

        Args:
            url: 文件URL
            output_path: 保存路径
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    raise Exception(f"下载失败: HTTP {response.status}")

                async with aiofiles.open(output_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        await f.write(chunk)

    async def download_tiktok_video(self,aweme_id: str, output_dir: str) -> str:
        """
        下载TikTok视频

        Args:
            aweme_id: TikTok视频ID
            output_dir: 输出目录

        Returns:
            下载的视频文件路径
        """
        logger.info(f"下载TikTok视频: {aweme_id}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        output_path = os.path.join(output_dir, f"tiktok_{aweme_id}_{file_id}.mp4")

        try:
            # 获取TikTok无水印链接
            video_url = await self.get_tiktok_download_url(aweme_id)

            # 下载视频
            await self.download_file(video_url, output_path)

            logger.info(f"TikTok视频已下载: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"下载TikTok视频失败: {str(e)}", exc_info=True)
            raise Exception(f"下载TikTok视频失败: {str(e)}")

    async def get_tiktok_download_url(self,aweme_id: str) -> dict[str, str | None] | Any:
        """
        获取TikTok视频的无水印下载链接

        Args:
            aweme_id: TikTok视频ID

        Returns:
            无水印视频URL
        """


        logger.info(f"🔍 正在获取视频数据: {aweme_id}...")

        video_crawler = VideoCollector(self.tikhub_api_key)
        video_data = await video_crawler.collect_single_video(aweme_id)

        if not video_data.get('video'):
            logger.warning(f"❌ 未找到视频数据: {aweme_id}")
            return {
                'aweme_id': aweme_id,
                'video': None,
                'timestamp': datetime.now().isoformat()
            }
        video_cleaner = VideoCleaner()
        cleaned_video_data = await video_cleaner.clean_single_video(video_data['video'])
        video_url = cleaned_video_data['video']['share_url']


        logger.info(f"✅ 已获取视频url数据: {video_url}")
        return video_url

    async def run_command(self,cmd: List[str]) -> str:
        """异步运行命令并返回stdout"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"命令执行失败 (返回码 {process.returncode}): {stderr.decode()}")
            raise Exception(f"命令执行失败: {stderr.decode()}")

        return stdout.decode()

    async def extract_audio(self, video_path: str) -> str:
        """从视频中提取音频为WAV格式"""
        # 使用临时文件
        temp_audio = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # 禁用视频
            "-acodec", "pcm_s16le",  # 16位PCM编码
            "-ar", "16000",  # 16kHz采样率
            "-ac", "1",  # 单声道
            "-y",  # 覆盖输出文件
            temp_audio
        ]

        await self.run_command(cmd)
        return temp_audio

    async def recognize_speech(self,video_path: str, language: str = "auto") -> str:
        """
        识别视频中的语音

        Args:
            video_path: 视频文件路径
            language: 语言代码

        Returns:
            转录的文本
        """
        logger.info(f"为视频 {video_path} 识别语音 (语言: {language})")

        # 提取音频
        audio_path = await self.extract_audio(video_path)

        try:
            # 尝试使用本地Whisper模型 (如果有安装)
            transcript = await self.recognize_with_whisper(audio_path, language)
            if transcript:
                return transcript
        except Exception as e:
            logger.warning(f"Whisper识别失败: {str(e)}")


        finally:
            # 清理临时文件
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {str(e)}")

    async def recognize_with_whisper(self, audio_path: str, language: str = "auto") -> Optional[str]:
        """
        使用OpenAI Whisper模型进行语音识别

        Args:
            audio_path: 音频文件路径
            language: 语言代码

        Returns:
            转录的文本，失败则返回None
        """
        try:

            wisper = WhisperLemonFox()
            whisper_result = await wisper.transcriptions(
                file=audio_path,
                response_format="verbose_json",
                speaker_labels=False,
                prompt="",
                language="",
                callback_url="",
                translate=False,
                timestamp_granularities=None,
                timeout=60
            )
            return whisper_result.get("text", "")



        except ImportError:
            logger.warning("未安装Whisper模块")
            return None
        except Exception as e:
            logger.error(f"Whisper识别失败: {str(e)}", exc_info=True)
            return None


    async def translate_text(self,text: str, source_language: str = "auto", target_language: str = "en",
                            ) -> str:
        """
        翻译文本

        Args:
            text: 需要翻译的文本
            source_language: 源语言代码
            target_language: 目标语言代码


        Returns:
            翻译后的文本
        """
        logger.info(f"翻译文本 ({source_language} -> {target_language}) ")

        if source_language == target_language:
            logger.info("源语言和目标语言相同，无需翻译")
            return text
        try:

            return await self.translate_with_chagpt(text, source_language, target_language)


        except Exception as e:
            logger.error(f"翻译失败: {str(e)}", exc_info=True)
            # 返回原文，避免完全失败
            return text

    async def translate_with_chagpt(self,text: str, source_lang: str, target_lang: str) -> str:
        system_prompt = "你是一位专业的翻译助手，能够准确地将任何语言翻译成目标语言。"
        user_prompt = (f"请将以下{source_lang}文本翻译为{target_lang}：\n"
                       f"{text}")
        chat_translate = ChatGPT()
        translated_text = await chat_translate.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="",
            temperature=0.7,
            max_tokens=10000,
            timeout=60,
        )

        return translated_text["choices"][0]["message"]["content"].strip()


class CleanupService:
    """管理静态文件清理的服务"""

    def __init__(self, directories, interval=3600):
        """
        初始化清理服务

        Args:
            directories: 需要清理的目录或目录列表
            interval: 清理间隔（秒），默认为1小时
        """
        if isinstance(directories, str):
            self.directories = [directories]
        else:
            self.directories = directories

        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        """启动清理服务"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._cleanup_loop)
        self.thread.daemon = True
        self.thread.start()

        logger.info(f"文件清理服务已启动，间隔: {self.interval}秒")

    def stop(self):
        """停止清理服务"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        logger.info("文件清理服务已停止")

    def _cleanup_loop(self):
        """清理循环"""
        while self.running:
            try:
                self._cleanup_files()
            except Exception as e:
                logger.error(f"文件清理过程中出错: {str(e)}", exc_info=True)

            # 休眠一段时间
            for _ in range(int(self.interval / 10)):
                if not self.running:
                    break
                time.sleep(10)

    def _cleanup_files(self):
        """清理过期文件"""
        current_time = time.time()
        expiration_time = current_time - self.interval

        for directory in self.directories:
            if not os.path.exists(directory):
                continue

            logger.info(f"清理目录: {directory}")
            cleanup_count = 0

            for file_path in Path(directory).glob('*'):
                if not file_path.is_file():
                    continue

                # 获取文件修改时间
                mod_time = os.path.getmtime(file_path)

                # 如果文件超过保留时间，则删除
                if mod_time < expiration_time:
                    try:
                        os.remove(file_path)
                        cleanup_count += 1
                    except Exception as e:
                        logger.error(f"删除文件 {file_path} 失败: {str(e)}")

            logger.info(f"已清理 {cleanup_count} 个文件")


def cleanup_temp_files(max_age=3600):
    """
    清理临时文件

    Args:
        max_age: 最大文件保留时间（秒）
    """
    import tempfile

    temp_dir = tempfile.gettempdir()
    current_time = time.time()
    expiration_time = current_time - max_age

    logger.info(f"清理临时目录: {temp_dir}")
    cleanup_count = 0

    # 清理临时目录中的视频处理相关文件
    for pattern in ['*.wav', '*.mp4', '*.srt', '*.json']:
        for file_path in Path(temp_dir).glob(pattern):
            try:
                # 检查文件修改时间
                if os.path.getmtime(file_path) < expiration_time:
                    os.remove(file_path)
                    cleanup_count += 1
            except Exception as e:
                logger.error(f"删除临时文件 {file_path} 失败: {str(e)}")

    logger.info(f"已清理 {cleanup_count} 个临时文件")


class JobStatus(str, Enum):
    """作业状态枚举"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobManager:
    """管理异步作业状态和结果"""

    def __init__(self):
        """初始化作业管理器"""
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def create_job(self, job_id: str, parent_job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        创建新作业

        Args:
            job_id: 作业ID
            parent_job_id: 父作业ID (用于批处理中的子任务)

        Returns:
            新创建的作业信息
        """
        with self.lock:
            job_info = {
                "job_id": job_id,
                "status": JobStatus.QUEUED,
                "message": "任务已加入队列",
                "created_at": time.time(),
                "updated_at": time.time(),
                "result": None,
                "parent_job_id": parent_job_id
            }

            self.jobs[job_id] = job_info
            logger.info(f"创建作业: {job_id}")
            return job_info

    def update_job(self, job_id: str, status: JobStatus, message: str, result: Optional[Dict] = None) -> Optional[
        Dict[str, Any]]:
        """
        更新作业状态

        Args:
            job_id: 作业ID
            status: 新状态
            message: 状态消息
            result: 作业结果 (如果有)

        Returns:
            更新后的作业信息，如果作业不存在则返回None
        """
        with self.lock:
            if job_id not in self.jobs:
                logger.warning(f"尝试更新不存在的作业: {job_id}")
                return None

            job_info = self.jobs[job_id]
            job_info["status"] = status
            job_info["message"] = message
            job_info["updated_at"] = time.time()

            if result is not None:
                job_info["result"] = result

            # 如果有父作业，检查是否需要更新父作业状态
            if job_info.get("parent_job_id"):
                self._update_parent_job_status(job_info["parent_job_id"])

            logger.info(f"更新作业 {job_id}: {status} - {message}")
            return job_info

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        获取作业信息

        Args:
            job_id: 作业ID

        Returns:
            作业信息，如果作业不存在则返回None
        """
        with self.lock:
            if job_id not in self.jobs:
                return None
            return self.jobs[job_id].copy()

    def list_jobs(self, status: Optional[str] = None, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """
        列出作业

        Args:
            status: 可选的状态过滤
            limit: 返回结果数量限制
            skip: 跳过结果数量

        Returns:
            作业列表
        """
        with self.lock:
            # 过滤作业
            filtered_jobs = list(self.jobs.values())

            if status:
                filtered_jobs = [job for job in filtered_jobs if job["status"] == status]

            # 按创建时间排序 (最新的在前)
            filtered_jobs.sort(key=lambda x: x["created_at"], reverse=True)

            # 分页
            return filtered_jobs[skip:skip + limit]

    def delete_job(self, job_id: str) -> bool:
        """
        删除作业

        Args:
            job_id: 作业ID

        Returns:
            是否成功删除
        """
        with self.lock:
            if job_id not in self.jobs:
                return False

            del self.jobs[job_id]
            logger.info(f"删除作业: {job_id}")
            return True

    def cleanup_old_jobs(self, max_age_seconds: int = 86400) -> int:
        """
        清理旧作业

        Args:
            max_age_seconds: 最大作业保留时间(秒)，默认为1天

        Returns:
            清理的作业数量
        """
        current_time = time.time()
        to_delete = []

        with self.lock:
            for job_id, job_info in self.jobs.items():
                job_age = current_time - job_info["created_at"]
                if job_age > max_age_seconds:
                    to_delete.append(job_id)

            for job_id in to_delete:
                del self.jobs[job_id]

        logger.info(f"清理了 {len(to_delete)} 个旧作业")
        return len(to_delete)

    def _update_parent_job_status(self, parent_job_id: str) -> None:
        """
        更新父作业状态（基于所有子作业状态）

        Args:
            parent_job_id: 父作业ID
        """
        if parent_job_id not in self.jobs:
            return

        # 获取所有子作业
        child_jobs = [
            job for job in self.jobs.values()
            if job.get("parent_job_id") == parent_job_id
        ]

        if not child_jobs:
            return

        # 统计各状态子作业数量
        total_jobs = len(child_jobs)
        completed_jobs = sum(1 for job in child_jobs if job["status"] == JobStatus.COMPLETED)
        failed_jobs = sum(1 for job in child_jobs if job["status"] == JobStatus.FAILED)
        processing_jobs = sum(1 for job in child_jobs if job["status"] == JobStatus.PROCESSING)

        parent_job = self.jobs[parent_job_id]

        # 更新父作业状态
        if completed_jobs + failed_jobs == total_jobs:
            # 所有子作业已完成或失败
            if failed_jobs == 0:
                parent_job["status"] = JobStatus.COMPLETED
                parent_job["message"] = f"所有 {total_jobs} 个任务已完成"
            elif completed_jobs == 0:
                parent_job["status"] = JobStatus.FAILED
                parent_job["message"] = f"所有 {total_jobs} 个任务失败"
            else:
                parent_job["status"] = JobStatus.COMPLETED
                parent_job[
                    "message"] = f"部分完成: {completed_jobs}/{total_jobs} 个任务成功, {failed_jobs}/{total_jobs} 个任务失败"
        elif processing_jobs > 0:
            # 有子作业正在处理
            progress = int((completed_jobs + failed_jobs) / total_jobs * 100)
            parent_job["status"] = JobStatus.PROCESSING
            parent_job["message"] = f"进行中: {completed_jobs + failed_jobs}/{total_jobs} 个任务已处理 ({progress}%)"

        parent_job["updated_at"] = time.time()


# 常量定义
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"


class Subtitle:
    """字幕类，表示一条字幕"""

    def __init__(self, index: int, start_time: float, end_time: float, text: str):
        self.index = index
        self.start_time = start_time  # 秒
        self.end_time = end_time  # 秒
        self.text = text

    def format_time(self, time_in_seconds: float) -> str:
        """将秒转换为SRT格式的时间字符串 (HH:MM:SS,mmm)"""
        td = timedelta(seconds=time_in_seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def to_srt(self) -> str:
        """转换为SRT格式字符串"""
        start_str = self.format_time(self.start_time)
        end_str = self.format_time(self.end_time)
        return f"{self.index}\n{start_str} --> {end_str}\n{self.text}\n"


async def run_command(cmd: List[str]) -> Tuple[str, str]:
    """异步运行命令并返回stdout和stderr"""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return stdout.decode(), stderr.decode()


async def get_video_info(video_path: str) -> Dict[str, Any]:
    """获取视频信息"""
    cmd = [
        FFPROBE_BIN,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    stdout, stderr = await run_command(cmd)
    if stderr:
        logger.warning(f"FFprobe stderr: {stderr}")

    info = json.loads(stdout)
    return info


async def text_to_subtitles(text: str, duration: float, max_words_per_line: int = 7) -> List[Subtitle]:
    """
    将文本转换为字幕列表

    Args:
        text: 文本内容
        duration: 视频时长(秒)
        max_words_per_line: 每行最大单词数

    Returns:
        字幕列表
    """
    # 清理文本，并按句子拆分
    clean_text = re.sub(r'\s+', ' ', text).strip()

    # 按标点符号分句
    sentences = re.split(r'(?<=[.!?;。！？；])\s*', clean_text)
    sentences = [s for s in sentences if s.strip()]

    # 估计每个单词的平均时长
    words = clean_text.split()
    total_words = len(words)
    avg_word_duration = duration / max(total_words, 1)

    # 构建字幕
    subtitles = []
    index = 1
    current_time = 0.0

    for sentence in sentences:
        # 拆分句子成多行
        words_in_sentence = sentence.split()

        for i in range(0, len(words_in_sentence), max_words_per_line):
            chunk = ' '.join(words_in_sentence[i:i + max_words_per_line])
            if not chunk:
                continue

            word_count = len(chunk.split())
            chunk_duration = word_count * avg_word_duration

            # 确保每条字幕至少显示1秒
            chunk_duration = max(chunk_duration, 1.0)

            end_time = min(current_time + chunk_duration, duration)

            subtitle = Subtitle(index, current_time, end_time, chunk)
            subtitles.append(subtitle)

            current_time = end_time
            index += 1

            # 如果已经到达视频结尾，则停止
            if current_time >= duration:
                break

        # 在句子之间添加一点间隔
        current_time = min(current_time + 0.2, duration)

    return subtitles


async def write_srt_file(subtitles: List[Subtitle], output_path: str) -> None:
    """写入SRT文件"""
    srt_content = ""
    for subtitle in subtitles:
        srt_content += subtitle.to_srt() + "\n"

    async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
        await f.write(srt_content)


async def generate_subtitles(video_path: str, transcript: str, output_dir: str, subtitle_format: str = "srt") -> Tuple[
    str, str]:
    """
    生成字幕并添加到视频

    Args:
        video_path: 视频文件路径
        transcript: 字幕文本内容
        output_dir: 输出目录
        subtitle_format: 字幕格式

    Returns:
        (输出视频路径, SRT文件路径)
    """
    logger.info(f"为视频 {video_path} 生成字幕")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成唯一文件名
    file_id = str(uuid.uuid4())
    base_name, ext = os.path.splitext(os.path.basename(video_path))
    output_srt = os.path.join(output_dir, f"{base_name}_{file_id}.srt")
    output_video = os.path.join(output_dir, f"{base_name}_{file_id}_subtitled{ext}")

    # 获取视频信息
    video_info = await get_video_info(video_path)
    duration = float(video_info.get('format', {}).get('duration', 0))

    # 将文本转换为字幕
    subtitles = await text_to_subtitles(transcript, duration)

    # 写入SRT文件
    await write_srt_file(subtitles, output_srt)

    # 使用FFmpeg将字幕添加到视频
    cmd = [
        FFMPEG_BIN,
        "-i", video_path,
        "-i", output_srt,
        "-map", "0",
        "-map", "1",
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        "-y",
        output_video
    ]

    stdout, stderr = await run_command(cmd)
    if stderr and "Error" in stderr:
        raise Exception(f"FFmpeg错误: {stderr}")

    logger.info(f"字幕视频已生成: {output_video}")
    return output_video, output_srt


async def extract_subtitles(video_path: str, output_dir: str) -> Tuple[str, str]:
    """
    从视频中提取字幕

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录

    Returns:
        (提取的字幕文本, SRT文件路径)
    """
    logger.info(f"从视频 {video_path} 提取字幕")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成唯一文件名
    file_id = str(uuid.uuid4())
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_srt = os.path.join(output_dir, f"{base_name}_{file_id}_extracted.srt")

    # 使用FFmpeg提取字幕
    cmd = [
        FFMPEG_BIN,
        "-i", video_path,
        "-map", "0:s:0",
        "-y",
        output_srt
    ]

    stdout, stderr = await run_command(cmd)

    # 如果没有嵌入式字幕，尝试使用OCR识别硬编码字幕
    if "Stream map" in stderr and "matches no streams" in stderr:
        logger.info("没有嵌入式字幕，尝试OCR识别硬编码字幕")
        subtitle_text, output_srt = await ocr_extract_subtitles(video_path, output_dir)
        return subtitle_text, output_srt

    # 读取SRT文件
    subtitle_text = ""
    if os.path.exists(output_srt):
        async with aiofiles.open(output_srt, 'r', encoding='utf-8') as f:
            content = await f.read()
            # 提取纯文本（去除时间码和编号）
            lines = content.split('\n')
            for i in range(len(lines)):
                if not re.match(r'^\d+$', lines[i]) and not re.match(
                        r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$', lines[i]):
                    if lines[i].strip():
                        subtitle_text += lines[i] + " "

    return subtitle_text.strip(), output_srt


async def ocr_extract_subtitles(video_path: str, output_dir: str) -> Tuple[str, str]:
    """
    使用OCR提取硬编码字幕

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录

    Returns:
        (SRT文件路径, 提取的字幕文本)
    """
    # 这里实现OCR字幕提取逻辑
    # 注意: 实际实现需要集成OCR库，如Tesseract或云OCR服务

    file_id = str(uuid.uuid4())
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_srt = os.path.join(output_dir, f"{base_name}_{file_id}_ocr.srt")

    ocr = VideoOCR(['en', 'ch_sim'])

    try:
        # Analyze video (every 30 frames)
        results = await ocr.analyze_video(
            video_path,
            # "https://v45-p.tiktokcdn-us.com/900115978f53c21ad336dbb55adc4b2b/67c1b774/video/tos/useast5/tos-useast5-ve-0068c001-tx/oMBnVfDKvySfuWOY9uuPSulEgugFQCncDIgA7U/?a=1233&bti=OUBzOTg7QGo6OjZAL3AjLTAzYCMxNDNg&ch=0&cr=13&dr=0&er=0&lr=all&net=0&cd=0%7C0%7C0%7C&br=3014&bt=1507&cs=0&ds=6&ft=yh7iX9DfxxOusQOFDnL76GFpA-JuGb1nNADwF_utoFmQ2Nz7T&mime_type=video_mp4&qs=0&rc=aGVkOjo8NjY5MzU3NDdpPEBpamRqa2o5cjxseDMzZzgzNEAvYV4yXzMtNS0xNl5hXi1iYSNfcHFfMmRja3JgLS1kLy9zcw%3D%3D&vvpl=1&l=20250228071638948B86E678A8EC05FCB1&btag=e00095000",
            time_interval=90,  # 每30帧分析一次
            confidence_threshold=0.5
        )
        subtitle_text = ""
        for result in results:
            # 正确地遍历 'texts' 列表并提取 'text' 字段
            texts = [text_dict['text'] for text_dict in result['texts']]
            subtitle_text += " ".join(texts) + " "

        await ocr.save_analysis(results, output_srt)
        return subtitle_text, output_srt

    except Exception as e:
        print(f"Error: {str(e)}")


async def remove_subtitles(video_path: str, output_dir: str) -> str:
    """
    移除视频中的硬编码字幕

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录

    Returns:
        输出视频路径
    """
    logger.info(f"从视频 {video_path} 移除字幕")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成唯一文件名
    file_id = str(uuid.uuid4())
    base_name, ext = os.path.splitext(os.path.basename(video_path))
    print(base_name)
    output_video = os.path.join(output_dir, f"{base_name}_{file_id}_nosubtitles{ext}")
    print(output_video)

    # 使用FFmpeg移除硬编码字幕
    # 注意：实际移除硬编码字幕需要使用复杂的视频处理技术
    # 这里使用一个FFmpeg滤镜作为简单示例，实际效果可能不理想
    cmd = [
        FFMPEG_BIN,
        "-i", video_path,
        "-c", "copy",  # 复制而不是重新编码
        "-map", "0:v",  # 映射输入文件的所有视频流
        "-map", "0:a",  # 映射输入文件的所有音频流
        "-sn",  # 不包含字幕流
        "-y",  # 覆盖输出文件（如果存在）
        output_video
    ]

    stdout, stderr = await run_command(cmd)

    logger.info(f"字幕已移除: {output_video}")
    return output_video


class FileWrapper:
    def __init__(self, filepath):
        # 保存原始文件路径
        self._filepath = filepath
        # 提取文件名（带扩展名）
        self.filename = os.path.basename(filepath)

    async def read(self):
        # 每次读取时打开文件（二进制模式）
        with open(self._filepath, "rb") as f:
            return f.read()







