# -*- coding: utf-8 -*-
"""
wf_dump_long.py - 以 20 秒运行 wf_dump.py

更长的窗口便于看清 RR 序列的抖动/漏搏模式。

用法:
    mpremote connect COM6 run wf_dump_long.py
"""

DUMP_SEC = 20
exec(open("wf_dump.py").read())
