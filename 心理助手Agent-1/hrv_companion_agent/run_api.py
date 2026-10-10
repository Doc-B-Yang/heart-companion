# -*- coding: utf-8 -*-
"""
API 服务启动脚本
================
用法:
  python run_api.py            # 默认 http://127.0.0.1:8000
  python run_api.py --port 9000 --reload

启动后:
  API 文档: http://127.0.0.1:8000/docs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'硬件开发'/'.runtime'))


def main():
    parser = argparse.ArgumentParser(description="启动主动式情感陪伴系统 API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn
    print("🚀 启动 主动式情感陪伴系统 API ...")
    print(f"   📖 文档: http://{args.host}:{args.port}/docs")
    uvicorn.run("hrv_companion_agent.backend.main:app",
                host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
