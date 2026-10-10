# -*- coding: utf-8 -*-
"""
finger_test_long.py - 以 90 秒编排运行 finger_test.py

编排:
    0 ~ 25s   静止
   25 ~ 55s   晃动 (轻轻晃动手 / 轻捏传感器)
   55 ~ 90s   恢复静止

为什么用启动器: `mpremote run` 不支持传脚本参数, 且 MicroPython 的
ESP32 端口没有 sys.argv。exec 共用调用方的 globals, 所以先注入再 exec。

用法:
    mpremote connect COM6 run finger_test_long.py
"""

FINGER_SEC = 90
STILL1 = 25
SHAKE_B = 55
exec(open("finger_test.py").read())
