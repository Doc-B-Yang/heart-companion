# -*- coding: utf-8 -*-
"""
wf_dump_15.py - 以 15 秒运行 wf_dump.py

更长的窗口便于判别采样节奏与 RR 抖动模式, 并把完整波形落盘供离线复算。

用法:
    mpremote connect COM6 run wf_dump_15.py
"""

DUMP_SEC = 15
exec(open("wf_dump.py").read())
