# -*- coding: utf-8 -*-
"""
mpu_test_quick.py - 以 quick 模式运行 mpu_test.py (跳过实时监视, 约 10 秒)

硬件体检用: 只做 I2C 扫描 / 芯片识别 / 寄存器校验 / 静态质量 / INT 通路,
不做 12 秒实时监视。适合每次插拔接线后快速复验。

用法:
    mpremote connect COM6 run mpu_test_quick.py
"""

MPU_MODE = "quick"
exec(open("mpu_test.py").read())
