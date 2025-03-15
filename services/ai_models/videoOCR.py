# -*- coding: utf-8 -*-
"""
@file: videoOCR.py
@desc: 处理TikTok评论的代理类，提供评论获取、分析和潜在客户识别功能
@auth: Callmeiks
"""
import math

import cv2
import easyocr
import numpy as np
from typing import List, Dict, Optional, Union
import asyncio
from urllib.parse import urlparse
import warnings

from app.utils.logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

warnings.filterwarnings('ignore')


class VideoOCR:
    """
    视频OCR类，用于从视频中提取文本内容。
    支持本地文件和URL。
    """

    def __init__(self, languages: List[str] = ['en', 'ch_sim']):
        """
        初始化OCR读取器。

        Args:
            languages: OCR识别的语言列表
        """

        self.reader = easyocr.Reader(languages)

    def _is_url(self, path: str) -> bool:
        """检查路径是否为URL。"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False

    async def _process_frame(self, frame) -> List[Dict]:
        """处理单帧图像以提取文本内容。"""
        try:
            results = self.reader.readtext(frame)

            texts = []
            for bbox, text, conf in results:
                bbox = [[int(point) for point in pos] for pos in bbox]
                texts.append({
                    'text': text,
                    'confidence': round(conf, 3),
                    'position': bbox
                })

            return texts

        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return []

    async def analyze_video(self,
                            video_path: str,
                            time_interval: float = 3.0,
                            confidence_threshold: float = 0.5) -> List[Dict]:
        """
        从视频中提取文本内容，按照「固定秒数间隔」来采样帧。

        Args:
            video_path: 视频文件路径
            time_interval: 分析帧的时间间隔（秒）
            confidence_threshold: 文本识别的置信度阈值

        Returns:
            分析结果列表，每项包含:
              - frame_number:    帧号
              - timestamp:       时间戳(秒)
              - texts:          识别到的文本及其置信度等信息
        """

        logger.info(f"Opening video: {video_path}")
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # 视频总时长
            duration = total_frames / fps if fps > 0 else 0

            logger.info(f"视频信息:")
            logger.info(f"- FPS: {fps}")
            logger.info(f"- 总帧数: {total_frames}")
            logger.info(f"- 视频时长: {duration:.2f}秒")
            logger.info(f"- 时间间隔: {time_interval}秒")
            logger.info(f"- 置信度阈值: {confidence_threshold}")

            results = []
            current_time = 0.0
            frames_analyzed = 0

            while current_time <= duration:
                # 根据时间戳跳转到指定位置（毫秒）
                video.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                success, frame = video.read()

                if not success:
                    # 已到达视频末尾或其他读取失败情况
                    break

                # 计算当前帧号（向下取整更合理，也可做四舍五入）
                frame_number = math.floor(video.get(cv2.CAP_PROP_POS_FRAMES))

                try:
                    # 自定义的文本处理逻辑，如OCR等
                    texts = await self._process_frame(frame)
                    texts = [t for t in texts if t['confidence'] >= confidence_threshold]

                    if texts:
                        result = {
                            'frame_number': frame_number,
                            'timestamp': round(current_time, 2),
                            'texts': texts
                        }
                        results.append(result)

                    frames_analyzed += 1
                    progress = (current_time / duration * 100) if duration > 0 else 0
                    logger.info(
                        f"\r正在分析第{frames_analyzed}处时间({current_time:.2f}s)"
                        f" - 进度: {progress:.1f}%",
                    )

                except Exception as e:
                    logger.error(f"\n分析第{frames_analyzed}次({current_time:.2f}s)时出错: {str(e)}")

                # 前进到下一个时间点
                current_time += time_interval

            logger.info(f"视频分析完成，共分析{frames_analyzed}个时间点")
            return results

        finally:
            video.release()

    async def save_analysis(self,
                            results: List[Dict],
                            output_path: str,
                            format: str = 'text'):
        """Save analysis results."""
        import json

        try:
            if not results:
                print("No results to save!")
                return

            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            else:
                text_lines = []
                for result in results:
                    frame = result['frame_number']
                    timestamp = result['timestamp']
                    texts = [f"- {t['text']} (confidence: {t['confidence']})"
                             for t in result['texts']]

                    text_lines.append(f"\nFrame {frame} (Time: {timestamp}s)")
                    text_lines.extend(texts)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(text_lines))

            print(f"Results saved to: {output_path}")

        except Exception as e:
            print(f"Error saving results: {str(e)}")


# Example usage
async def main():
    # Initialize
    ocr = VideoOCR(['en', 'ch_sim'])

    try:
        # Analyze video (every 30 frames)
        results = await ocr.analyze_video(
            "https://v45-p.tiktokcdn-us.com/900115978f53c21ad336dbb55adc4b2b/67c1b774/video/tos/useast5/tos-useast5-ve-0068c001-tx/oMBnVfDKvySfuWOY9uuPSulEgugFQCncDIgA7U/?a=1233&bti=OUBzOTg7QGo6OjZAL3AjLTAzYCMxNDNg&ch=0&cr=13&dr=0&er=0&lr=all&net=0&cd=0%7C0%7C0%7C&br=3014&bt=1507&cs=0&ds=6&ft=yh7iX9DfxxOusQOFDnL76GFpA-JuGb1nNADwF_utoFmQ2Nz7T&mime_type=video_mp4&qs=0&rc=aGVkOjo8NjY5MzU3NDdpPEBpamRqa2o5cjxseDMzZzgzNEAvYV4yXzMtNS0xNl5hXi1iYSNfcHFfMmRja3JgLS1kLy9zcw%3D%3D&vvpl=1&l=20250228071638948B86E678A8EC05FCB1&btag=e00095000",
            time_interval=90,  # 每30帧分析一次
            confidence_threshold=0.5
        )

        await ocr.save_analysis(results, "ocr_results.txt")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())