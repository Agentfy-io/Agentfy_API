import os
import time
import threading
from app.utils.logger import setup_logger
from pathlib import Path

logger = setup_logger(__name__)

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
