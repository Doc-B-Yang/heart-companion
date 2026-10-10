# -*- coding: utf-8 -*-
"""
后端服务入口 —— FastAPI
========================
对应计划书 §4.4:
  - BLE数据接收 / 情绪状态查询 / 主动触发检查 / AI对话(含WebSocket流式)
  - 用户画像 / TTS语音合成 / 干预反馈 / AB测试辅助

启动: 在项目根目录运行 `python run_api.py`（自动注入路径并启动 uvicorn）
文档: 启动后访问 http://127.0.0.1:8000/docs
"""

import sys
import os
from pathlib import Path

# 保证 hrv_companion_agent 包可导入（无论从何处启动）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routes import router  # noqa: E402
from .. import config  # noqa: E402

app = FastAPI(
    title="主动式情感陪伴系统 API",
    description="基于实时心率变异性驱动的主动式情感陪伴系统 —— 后端服务",
    version="3.0.0",
)

# CORS: 允许 Flutter APP / Web 前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.WEB_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
WEB_DIR = Path(__file__).resolve().parent / 'web'
app.mount('/static',StaticFiles(directory=WEB_DIR),name='static')


@app.get("/")
def root():
    return FileResponse(WEB_DIR/'index.html')
