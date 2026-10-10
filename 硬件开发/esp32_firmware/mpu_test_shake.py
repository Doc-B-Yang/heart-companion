# -*- coding: utf-8 -*-
"""
mpu_test_shake.py - 以 shake 模式运行 mpu_test.py (30 秒运动响应验证)

为什么需要这个文件:
    两个硬约束 ——
      ① `mpremote run` 不支持给脚本传命令行参数
         (`... run mpu_test.py --shake` 会把 --shake 当成 mpremote 的命令而报错);
      ② MicroPython 的 **ESP32 端口没有 sys.argv**。
    所以用一个启动器注入全局变量 MPU_MODE 后 exec 主脚本
    (exec 共用调用方的 globals, 主脚本因此能读到 MPU_MODE)。

用法:
    mpremote connect COM6 run mpu_test_shake.py

运行时请**持续轻晃 / 轻敲板子**, 直到 30 秒窗口结束。
"""

MPU_MODE = "shake"
exec(open("mpu_test.py").read())
