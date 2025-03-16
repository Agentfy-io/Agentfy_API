from services.ai_models.chatgpt import ChatGPT
from services.ai_models.claude import Claude
from typing import Dict, Any, List, Optional, Union
from types import SimpleNamespace
from app.utils.logger import setup_logger
from dotenv import load_dotenv
import shutil
from app.utils.job_manager import JobManager, JobStatus
import os
from app.utils.subtitle_generator import generate_subtitles, extract_subtitles, remove_subtitles, FileWrapper
from app.config import settings
from fastapi import HTTPException
import asyncio
import uuid
import aiofiles
from datetime import datetime
import aiohttp
from services.crawler.comment_crawler import VideoCollector, VideoCleaner
from services.ai_models.whisper import WhisperLemonFox
import tempfile

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






