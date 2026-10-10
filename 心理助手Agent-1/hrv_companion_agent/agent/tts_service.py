# -*- coding: utf-8 -*-
"""
TTS 语音合成服务
================
对应计划书 §4.3 tts_service.py / PPT STEP 05。

优先使用 Edge-TTS（微软免费神经网络语音，温暖人声）；未安装或未启用时
优雅降级（返回 None，仅文本交互）。所有网络依赖均可选，不影响主流程。
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from .. import config

logger = logging.getLogger("companion.tts")


def _edge_available() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


TTS_TIMEOUT_SEC = 8  # 合成超时（网络差时不阻塞对话）


async def synthesize(text: str, voice: Optional[str] = None,
                     out_path: Optional[str] = None) -> Optional[str]:
    """
    文本 → 语音文件（MP3）。返回文件路径；不可用时返回 None。

    Args:
        text: 待合成文本
        voice: Edge-TTS 音色（默认 zh-CN-XiaoyiNeural 温暖女声）
        out_path: 输出路径（默认 data/tts/<时间戳>.mp3）
    """
    if not config.TTS_ENABLED or not _edge_available():
        return None
    import edge_tts

    voice = voice or config.TTS_VOICE
    config.TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = out_path or str(config.TTS_OUTPUT_DIR / f"tts_{int(time.time() * 1000)}.mp3")
    try:
        communicate = edge_tts.Communicate(text, voice)
        # 超时保护: TTS 失败/卡顿不影响文本对话主流程
        await asyncio.wait_for(communicate.save(path), timeout=TTS_TIMEOUT_SEC)
        return path
    except Exception as e:  # noqa: BLE001
        logger.warning("TTS 合成失败或超时(%s)，本次仅文本回复", e)
        return None


def synthesize_sync(text: str, voice: Optional[str] = None,
                    out_path: Optional[str] = None) -> Optional[str]:
    """同步包装（CLI 使用）"""
    try:
        return asyncio.run(synthesize(text, voice, out_path))
    except RuntimeError:  # 已在事件循环中
        return None
