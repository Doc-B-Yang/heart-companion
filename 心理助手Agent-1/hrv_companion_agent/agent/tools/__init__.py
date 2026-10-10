# -*- coding: utf-8 -*-
"""
Agent 工具集
============
对应计划书 §4.3 tools/ 目录。

v2 工具矩阵（共 9 个，按「感知 → 决策 → 干预 → 记忆 → 转介」分层）:

===============  ==================  ================================
工具             层                  文献依据 / 职责
===============  ==================  ================================
emotion_tool     感知                文献1/2/3 —— 状态全貌（含质量与 PAD）
pad_tool         感知                文献2 —— PAD 三维 + 相对特征归因
comfort_tool     干预                技能匹配引擎（22 项循证技术）
breathing_guide  干预                呼吸生理（含 HRV 混杂提醒）
self_report_tool 校准                主观自评（大创实验真值来源）
memory_tool      记忆                文献5 —— FTA 情境单元写入
recall_tool      记忆                文献5/4 —— 检索与洞察
crisis_tool      安全                意念分级 + 安全计划 + 禁止清单
referral_tool    安全                分层转介 + 求助话术脚本
===============  ==================  ================================

设计原则:
  * **工具失败不得中断对话** —— 每个工具内部自行兜底，
    异常一律转成给模型的文字说明，而不是抛给主循环；
  * **每个工具的返回都带「表达纪律」** —— 约束模型不要复述数值、
    不要把工具输出念给用户。这是本项目 v2 的一贯做法：
    专业的判断留在系统内，对话里只呈现自然语言。
"""
from .base_tool import BaseTool, SimpleToolRegistry
from .emotion_tool import EmotionTool
from .pad_tool import PADTool
from .comfort_tool import (ComfortTool, SKILLS, get_skill, match_skills,
                           all_skill_ids)
from .breathing_guide import BreathingGuideTool
from .crisis_tool import CrisisTool, detect_level
from .memory_tool import MemoryTool
from .recall_tool import RecallTool
from .referral_tool import ReferralTool
from .self_report_tool import SelfReportTool

__all__ = [
    # 基类
    "BaseTool", "SimpleToolRegistry",
    # 感知
    "EmotionTool", "PADTool",
    # 干预
    "ComfortTool", "BreathingGuideTool",
    # 校准
    "SelfReportTool",
    # 记忆
    "MemoryTool", "RecallTool",
    # 安全
    "CrisisTool", "ReferralTool",
    # 技能库辅助
    "SKILLS", "get_skill", "match_skills", "all_skill_ids", "detect_level",
]
