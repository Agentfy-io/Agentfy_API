from enum import Enum
from typing import Dict, List, Optional, Any
import time
import threading
import json

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

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
    
    def update_job(self, job_id: str, status: JobStatus, message: str, result: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
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
            return filtered_jobs[skip:skip+limit]
    
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
                parent_job["message"] = f"部分完成: {completed_jobs}/{total_jobs} 个任务成功, {failed_jobs}/{total_jobs} 个任务失败"
        elif processing_jobs > 0:
            # 有子作业正在处理
            progress = int((completed_jobs + failed_jobs) / total_jobs * 100)
            parent_job["status"] = JobStatus.PROCESSING
            parent_job["message"] = f"进行中: {completed_jobs + failed_jobs}/{total_jobs} 个任务已处理 ({progress}%)"
        
        parent_job["updated_at"] = time.time()