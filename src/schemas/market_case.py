"""Pydantic models for Market Case — core daily data model of Daily Trading OS."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class FeaturesModel(BaseModel):
    dgs10: Optional[float] = None
    dgs2: Optional[float] = None
    vix: Optional[float] = None
    vix_chg: Optional[float] = None
    qqq_chg: Optional[float] = None
    smh_chg: Optional[float] = None
    nvda_chg: Optional[float] = None
    spy_chg: Optional[float] = None
    oil_chg: Optional[float] = None
    dxy_chg: Optional[float] = None
    breadth_proxy: Optional[float] = None
    qqq_gap: Optional[float] = None
    model_config = {"extra": "allow"}


class RegimeModel(BaseModel):
    label: str = "Unknown"
    confidence: float = 0.5


class HypothesisModel(BaseModel):
    id: str = ""
    statement: str = ""
    evidence: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    status: str = "待验证"  # 待验证 | 支持 | 削弱 | 推翻


class AttributionModel(BaseModel):
    ai: float = 0.0
    bond: float = 0.0
    oil: float = 0.0
    macro: float = 0.0
    other: float = 0.0


class LabelsModel(BaseModel):
    actual_driver: Optional[str] = None
    hypothesis_correct: Optional[str] = None  # 对 | 错 | 部分对


class MarketCaseModel(BaseModel):
    date: str
    features: FeaturesModel = Field(default_factory=FeaturesModel)
    regime: RegimeModel = Field(default_factory=RegimeModel)
    hypothesis: HypothesisModel = Field(default_factory=HypothesisModel)
    morning: dict[str, Any] = Field(default_factory=dict)
    intraday: dict[str, Any] = Field(default_factory=dict)
    trade_plan: dict[str, Any] = Field(default_factory=dict)
    actual: dict[str, Any] = Field(default_factory=dict)
    attribution: AttributionModel = Field(default_factory=AttributionModel)
    labels: LabelsModel = Field(default_factory=LabelsModel)
    surprise: Optional[str] = None
    lesson: Optional[str] = None
    playbook_refs: list[str] = Field(default_factory=list)
    bayesian_drivers: dict[str, float] = Field(default_factory=dict)
    data_gaps: list[str] = Field(default_factory=list)
