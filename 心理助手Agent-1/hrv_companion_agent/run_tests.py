# -*- coding: utf-8 -*-
"""
测试运行器
==========
用法:
  python run_tests.py            # 运行全部测试
  python run_tests.py -v         # 详细输出
"""

import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'硬件开发'/'.runtime'))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

if __name__ == "__main__":
    # Never write production memory or contact an LLM from unit tests.
    from hrv_companion_agent import config
    _test_data = tempfile.TemporaryDirectory(prefix='companion-tests-')
    config.DATA_DIR = Path(_test_data.name)
    config.LOG_DIR = config.DATA_DIR / 'logs'
    config.MEMORY_DB_PATH = Path(_test_data.name) / 'test-memory.db'
    config.LLM_API_KEY = ''
    config.TTS_ENABLED = False
    config.MULTIMODAL_MODEL_ENABLED = False
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(Path(__file__).parent / "tests"),
                            pattern="test_*.py")
    verbosity = 2 if "-v" in sys.argv else 1
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    _test_data.cleanup()
    sys.exit(0 if result.wasSuccessful() else 1)
