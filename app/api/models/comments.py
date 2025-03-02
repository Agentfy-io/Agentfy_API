from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime


class CommentBase(BaseModel):
    """基本评论模型"""
    comment_id: str = Field(..., description="评论ID")
    text: str = Field(..., description="评论内容")
    commenter_uniqueId: str = Field(..., description="评论者用户名")
    commenter_secuid: str = Field(..., description="评论者安全用户ID")


class Comment(CommentBase):
    """完整评论模型"""
    comment_language: Optional[str] = Field(None, description="评论语言")
    digg_count: int = Field(0, description="点赞数")
    reply_count: int = Field(0, description="回复数")
    commenter_region: Optional[str] = Field(None, description="评论者区域")
    ins_id: Optional[str] = Field(None, description="Instagram ID")
    twitter_id: Optional[str] = Field(None, description="Twitter ID")
    create_time: Optional[str] = Field(None, description="创建时间")


class VideoCommentsRequest(BaseModel):
    """获取视频评论请求模型"""
    aweme_id: str = Field(..., description="视频ID")

    @validator('aweme_id')
    def validate_aweme_id(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("aweme_id必须是非空字符串")
        return v


class VideoCommentsResponse(BaseModel):
    """视频评论响应模型"""
    aweme_id: str = Field(..., description="视频ID")
    comments: List[Comment] = Field(default_factory=list, description="评论列表")
    comment_count: int = Field(..., description="评论总数")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class AnalysisRequest(BaseModel):
    """评论分析请求基类"""
    aweme_id: str = Field(..., description="视频ID")
    batch_size: int = Field(30, description="每批处理的评论数量")


class PurchaseIntentRequest(AnalysisRequest):
    """购买意图分析请求模型"""
    pass


class PotentialCustomersRequest(AnalysisRequest):
    """潜在客户识别请求模型"""
    min_score: float = Field(50.0, description="最小参与度分数", ge=0.0, le=100.0)
    max_score: float = Field(100.0, description="最大参与度分数", ge=0.0, le=100.0)

    @validator('max_score')
    def validate_max_score(cls, v, values):
        if 'min_score' in values and v <= values['min_score']:
            raise ValueError("max_score必须大于min_score")
        return v


class SentimentDistribution(BaseModel):
    """情感分布模型"""
    counts: Dict[str, int] = Field(..., description="各情感类别的数量")
    percentages: Dict[str, float] = Field(..., description="各情感类别的百分比")


class PurchaseIntent(BaseModel):
    """购买意图统计模型"""
    total_comments: int = Field(..., description="评论总数")
    intent_count: int = Field(..., description="有购买意图的评论数")
    intent_rate: float = Field(..., description="购买意图比率")
    intent_by_interest_level: Dict[str, int] = Field(..., description="各兴趣水平的购买意图数量")


class InterestLevels(BaseModel):
    """兴趣水平统计模型"""
    counts: Dict[str, int] = Field(..., description="各兴趣水平的数量")
    percentages: Dict[str, float] = Field(..., description="各兴趣水平的百分比")


class PurchaseIntentAnalysisMeta(BaseModel):
    """购买意图分析元数据"""
    total_analyzed_comments: int = Field(..., description="分析的评论总数")
    aweme_id: str = Field(..., description="视频ID")
    analysis_type: str = Field(..., description="分析类型")
    analysis_timestamp: datetime = Field(..., description="分析时间戳")


class PurchaseIntentAnalysis(BaseModel):
    """购买意图分析结果模型"""
    sentiment_distribution: SentimentDistribution = Field(..., description="情感分布")
    purchase_intent: PurchaseIntent = Field(..., description="购买意图统计")
    interest_levels: InterestLevels = Field(..., description="兴趣水平统计")
    meta: PurchaseIntentAnalysisMeta = Field(..., description="元数据")


class PotentialCustomer(BaseModel):
    """潜在客户模型"""
    user: str = Field(..., description="用户名")
    potential_value: float = Field(..., description="潜在价值分数")
    secuid: str = Field(..., description="安全用户ID")
    text: str = Field(..., description="评论内容")


class PotentialCustomersAnalysis(BaseModel):
    """潜在客户分析结果模型"""
    aweme_id: str = Field(..., description="视频ID")
    total_potential_customers: int = Field(..., description="潜在客户总数")
    min_engagement_score: float = Field(..., description="最小参与度分数")
    max_engagement_score: float = Field(..., description="最大参与度分数")
    average_potential_value: float = Field(..., description="平均潜在价值")
    potential_customers: List[PotentialCustomer] = Field(..., description="潜在客户列表")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="分析时间戳")