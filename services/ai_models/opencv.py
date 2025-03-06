# -*- coding: utf-8 -*-
"""
@file: opencv.py
@desc: 视频图像处理类，用于从视频中提取图像并生成描述性文本。
@auth: Callmeiks
"""

import cv2
from transformers import pipeline
import math
from typing import List, Dict, Optional, Union
import os
import json
import asyncio
from urllib.parse import urlparse
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class OpenCV:
    """
    视频图像处理类，用于从视频中提取图像并生成描述性文本。
    支持本地文件和URL。
    """

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """初始化图像描述生成器。"""
        self.model_name = model_name
        self.captioner = pipeline("image-to-text", model=self.model_name)

    def _is_url(self, path: str) -> bool:
        """检查路径是否为URL。"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False

    async def _process_frame(self, frame) -> str:
        """处理单帧图像以生成描述性文本。"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # Generate caption
            caption = self.captioner(pil_image)[0]['generated_text']
            return caption
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return "Frame processing failed"

    async def analyze_video(self,
                            video_path: str,
                            frame_interval: float = 2.0) -> List[Dict]:
        """
        从视频中提取图像并生成描述性文本。

        Args:
            video_path: 视频文件路径
            frame_interval: 分析帧之间的间隔（秒）

        Returns:
            分析结果
        """

        print(f"Opening video: {video_path}")
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_step = int(frame_interval * fps)

            print(f"Video info - FPS: {fps}, Total frames: {total_frames}")

            results = []
            current_frame = 0
            frames_processed = 0

            while True:
                if frame_step > 1:
                    video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                success, frame = video.read()
                if not success:
                    break

                try:
                    caption = await self._process_frame(frame)

                    start_time = current_frame / fps
                    end_time = (current_frame + frame_step) / fps

                    result = {
                        'start_time': round(start_time, 1),
                        'end_time': round(end_time, 1),
                        'description': caption
                    }
                    results.append(result)

                    frames_processed += 1
                    progress = (current_frame / total_frames * 100) if total_frames > 0 else 0
                    print(f"\rProcessed frames: {frames_processed}, Progress: {progress:.1f}%", end='')

                except Exception as e:
                    print(f"\nError processing frame {frames_processed}: {str(e)}")

                current_frame += frame_step

            print("\nAnalysis complete!")
            return results

        finally:
            video.release()

    async def save_analysis(self,
                            results: List[Dict],
                            output_path: str,
                            format: str = 'text'):
        """
        保存分析结果到文件。

        Args:
            results: 分析结果
            output_path: 输出文件路径
            format: 输出格式（'text' 或 'json'）
        """

        try:
            if not results:
                print("No results to save!")
                return

            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            else:
                text_results = [
                    f"{res['start_time']}s-{res['end_time']}s, {res['description']}"
                    for res in results
                ]
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(text_results))
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")


async def main():

    opencv = OpenCV()

    try:
        # Analyze video
        results = await opencv.analyze_video(
            "https://example.com/video.mp4",
            frame_interval=2.0
        )

        # Save results
        await opencv.save_analysis(results, "output.txt")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())