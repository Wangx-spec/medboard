"""核心 pydantic 输出模型：三个 Agent 的结构化输出契约（M2 消费）+ M1 自检模型。

字段 description 会随 JSON schema 一起注入 prompt，是提示工程的一部分。
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class DifficultyAssessment(BaseModel):
    """M1 自检用：难度评估（也是分诊输出的子集）。"""

    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Question difficulty level"
    )
    justification: str = Field(..., description="Brief reasoning for the difficulty")


class ExpertProfile(BaseModel):
    """CV 式专家画像（机制来自 MedAgents-2：角色扮演提升推理质量）。"""

    name: str = Field(..., description="Name of the expert")
    past_experience: str = Field(
        ..., description="Detailed summary of clinical or research experience"
    )
    educational_background: str = Field(
        ..., description="Degrees, institutions, certifications, specialized training"
    )
    core_specialties: str = Field(
        ..., description="Key areas of expertise with precise descriptions"
    )


class TriageOutput(BaseModel):
    """分诊 Agent 输出：难度 + 专家面板 + 权重。"""

    difficulty: Literal["easy", "medium", "hard", "custom"] = Field(
        ..., description="Question difficulty level"
    )
    justification: str = Field(..., description="Reasoning for the difficulty level")
    specialties: List[str] = Field(..., description="Selected medical specialties")
    weights: List[float] = Field(
        ..., description="Weight for each expert, must sum to 1", min_length=1
    )
    job_titles: List[str] = Field(..., description="Unique job title for each expert")
    expert_profiles: List[ExpertProfile] = Field(
        ..., description="CV-style profile for each expert"
    )
    research_focuses: List[str] = Field(
        ..., description="Unique research focus area for each expert"
    )


class ExpertResponse(BaseModel):
    """专家 Agent 输出：答案 + 置信度 + 证据 + 理由。"""

    thought: str = Field(..., description="Chain-of-thought reasoning")
    answer: str = Field(
        ..., description="Selected multiple-choice answer key (A-Z)", pattern="^[A-Z]$"
    )
    evidences: List[str] = Field(
        default_factory=list, description="Evidence supporting the answer"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        ..., description="Confidence level in the answer"
    )
    justification: str = Field(..., description="Concise reasoning for the answer")


class ModeratorResponse(BaseModel):
    """主持人输出：轮次分析 + 早停决策。"""

    round_summary: str = Field(..., description="Overall summary of this round")
    key_insights: List[str] = Field(default_factory=list)
    areas_of_agreement: List[str] = Field(default_factory=list)
    areas_of_disagreement: List[str] = Field(default_factory=list)
    should_continue: bool = Field(
        ..., description="Whether discussion should continue to another round"
    )
    confidence_in_decision: Literal["low", "medium", "high"] = Field(
        ..., description="Confidence in the current leading answer"
    )
