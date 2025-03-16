import os
import uuid
import asyncio
import subprocess
import json
import re
from typing import Tuple, List, Dict, Optional, Any
from app.utils.logger import setup_logger
from datetime import timedelta
import aiofiles
from services.ai_models.videoOCR import VideoOCR

logger = setup_logger(__name__)

# 常量定义
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

class Subtitle:
    """字幕类，表示一条字幕"""
    def __init__(self, index: int, start_time: float, end_time: float, text: str):
        self.index = index
        self.start_time = start_time  # 秒
        self.end_time = end_time      # 秒
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
            chunk = ' '.join(words_in_sentence[i:i+max_words_per_line])
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

async def generate_subtitles(video_path: str, transcript: str, output_dir: str, subtitle_format: str = "srt") -> Tuple[str, str]:
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
                if not re.match(r'^\d+$', lines[i]) and not re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$', lines[i]):
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
