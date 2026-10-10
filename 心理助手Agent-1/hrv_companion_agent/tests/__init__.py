# -*- coding: utf-8 -*-
"""测试套件包（路径引导在 run_tests.py 与 unittest discover 两种方式下均生效）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
