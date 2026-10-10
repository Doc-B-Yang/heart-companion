# -*- coding: utf-8 -*-
"""
config.py - 硬件引脚与参数配置
项目: 基于实时心率变异性驱动的主动式情感陪伴系统
硬件: ESP32-WROOM-32E + MAX30102(PPG) + MPU6050(IMU) + ADS1115/GSR
      [LD2410D 雷达已于 2026-09-30 退役]

注意: 所有引脚均为 ESP32-WROOM-32E 的 GPIO 编号。
      LD2410D 默认波特率 = 115200 (切勿误用 LD2410B 的 256000)。
"""

# ==================== 状态指示 LED ====================
# 多数 ESP32 开发板板载 LED 在 GPIO2
LED_PIN = 2

# ==================== I2C: MAX30102 ====================
I2C_ID   = 0
I2C_SCL  = 22
I2C_SDA  = 21
I2C_FREQ = 400000               # 三驱动支持Fast-mode；部署后验证负载/读错
MAX30102_ADDR = 0x57            # 7-bit 从机地址

# MAX30102 采集参数
MAX30102_SAMPLE_RATE = 100     # 采样率 Hz (50/100/200/400/800/1000/1600/3200)
MAX30102_FIFO_AVG    = 1       # FIFO 平均 (1/2/4/8/16/32); =1 保留最大时间分辨率，利于 HRV
# ADC 满量程: 2048 / 4096 / 8192 / 16384 nA。
#   实测: 手指按压较紧时光电流很大, 4096nA 会直接削顶饱和(IR 恒为 262143),
#   饱和后波形被削平 → 峰值检测在不应期下限抖动 → HR 虚高。必须留足余量。
MAX30102_ADC_RANGE   = 16384

# LED 电流(起始值): 开机后由 MAX30102.auto_gain() 自适应调档,
#   目标 IR 直流区间 [25000, 100000] (18-bit 满量程 262143)。
#   实测: 手指贴合差异极大(跨 20 倍以上), 固定电流会过暗(SNR差)或饱和(削顶)。
LED_RED_CURRENT      = 0x1F    # 红光 LED 起始电流 (auto_gain 会覆盖)
LED_IR_CURRENT       = 0x1F    # 红外 LED 起始电流 (HRV 主用通道)

# ==================== 扩展模态器件 (2026-09-30 到货) ====================
# 全部挂在同一 I2C 总线(GPIO21/22)上, 三个地址不冲突。
#
# ⚠️ 供电铁律: 本组器件与 MAX30102 一律用 3.3V, 不要用 5V。
#    原因: GSR@5V 的 ADC 输出最高可达 5V, 超过 ADS1115 (VDD=3.3V) 的
#          绝对最大额定值 VDD+0.3V=3.6V, 会损坏其输入级;
#          MAX30102 模块接 5V 会把 I2C 上拉拉到 5V 电平, 损伤 ESP32 的 GPIO。

# --- ADS1115 16 位 ADC (GSR / 后续呼吸带的模拟前端) ---
# 接线: VDD→3V3, GND→GND, SDA→GPIO21, SCL→GPIO22, ADDR→GND(定址 0x48)
#       A0 ← GSR 的 ADC 输出
ADS1115_ADDR     = 0x48         # ADDR 接 GND=0x48 / VDD=0x49 / SDA=0x4A / SCL=0x4B
ADS1115_REG_CONV = 0x00         # 转换结果寄存器
ADS1115_REG_CFG  = 0x01         # 配置寄存器
# PGA 档位必须选 ±4.096V: GSR 输出 0~3.3V, 选 ±2.048V 会削顶, 选 ±6.144V 浪费分辨率
ADS1115_PGA      = 4.096        # 满量程电压(V); 对应配置字 0x0200
ADS1115_SPS      = 128          # 采样率(Hz): 8/16/32/64/128/250/475/860
GSR_ADC_CHANNEL  = 0            # GSR 接在 A0

# --- MPU6050 六轴 IMU (运动伪迹抑制) ---
# 接线: VCC→3V3, GND→GND, SDA→GPIO21, SCL→GPIO22, AD0→GND(定址 0x68)
#       XDA/XCL 悬空(仅用于外挂磁力计); INT 可选接 GPIO 用于 DMP 中断
MPU6050_ADDR     = 0x68         # AD0 接 GND=0x68 / 接 VDD=0x69
MPU6050_ENABLED  = True         # 未接 IMU 时置 False, 主程序自动跳过运动门控
# INT 引脚(可选, 轮询模式不需要; 仅 mpu_test.py 用它验证引脚通路)。
# 用户实接 GPIO17。雷达已退役(见 RADAR_ENABLED), GPIO17 不再冲突。
MPU6050_INT_PIN  = 17
# IMU 采样配置: 与 PPG 同频(主循环 10ms 一次 → 100Hz), 便于逐样本对齐运动标签
MPU6050_DLPF     = 3            # 数字低通 3 → 加速度 44Hz / 陀螺 42Hz (实测可用)
MPU6050_RATE_HZ  = 100
# 开机静止标定: 采陀螺零偏 + 重力方向参考 + |a| 基线。期间请勿触碰板子。
MPU6050_CALIB_SEC    = 2.0      # 陀螺零偏标定时长(秒)
MPU6050_GRAVITY_SEC  = 1.0      # 重力方向参考标定时长(秒)
# 运动伪迹抑制判据(三选一即判为运动, 该段 RR 剔除)
#   ① 加速度模长偏差: |  |a| - |g_ref|  | > MPU6050_MOTION_G
#   ② 角速度模长    : |ω - 零偏|             > MPU6050_GYRO_DPS
#   ③ 重力方向倾角  : 与参考方向的夹角       > MPU6050_TILT_DEG
#
# ⚠️【2026-09-30 实测得出, 关键】判据① 对**转动几乎不敏感** ——
#    旋转不改变加速度矢量的模长(重力大小恒为 1g)。实测用手晃板子时,
#    dev 多数只有 0.05~0.25 g(常低于阈值), 而 |ω| 高达 70~390 °/s。
#    手部抖动/腕带位移的主动成分正是**转动** ⇒ 只用判据① 会大面积漏检。
#    必须以 ②(或 ③) 为主, ① 只作为平移/碰撞的补充。
MPU6050_MOTION_G = 0.25         # 判据①阈值(单位 g); 实测静止噪声仅 0.0025 g
MPU6050_GYRO_DPS = 20.0         # 判据②阈值(°/s); 实测手部晃动 70~390 °/s
MPU6050_TILT_DEG = 8.0          # 判据③阈值(度); 与角速度互补, 不受零偏漂移影响

# --- GSR 皮肤电 (思知瑞 V2) ---
# 接线: VCC→3V3(必须), GND→GND, ADC→ADS1115 的 A0; BAT+/BAT- 不接
# 电极佩戴: 同一只手的食指 + 中指指腹, 静置 2~3 分钟后再开始记录
GSR_ENABLED      = True

# ==================== UART: LD2410D (已退役) ====================
# 【2026-09-30】雷达已从硬件上拆除。呼吸率改由 PPG 的 RSA(呼吸性窦性
#   心律不齐)在 PC 端 bridge.py 反推(零硬件), 存在检测改由 HRV 状态触发
#   —— 契合"呼吸率不离人、测量门槛低"的方案定案。
# 保留常量仅供参考; RADAR_ENABLED=False 时 main.py 不会初始化 UART,
#   也不会向 JSON 里写 radar 字段。
RADAR_ENABLED = False
RADAR_UART_ID = 2               # 使用 UART2
RADAR_TX      = 17              # ESP32 TX  -->  LD2410D RX (PIN3)
RADAR_RX      = 16              # ESP32 RX  <--  LD2410D TX (PIN2)
RADAR_BAUD    = 115200          # !!! LD2410D 固定 115200 (LD2410B 才是 256000)
RADAR_OUT_PIN = 4               # LD2410D PIN1 OUT (可选，高电平=有人)

# ==================== HRV 参数 ====================
HRV_WINDOW_SEC     = 60         # HRV 统计窗口(秒)
PPG_MIN_PEAK_MS    = 450        # 最小峰间隔(ms) → 上限 133 bpm
#   实测标定: 每个心搏主峰后 0.30~0.35s 常出现次峰(重搏波/反射波),
#   若不应期下限=300ms 则次峰被误检为心搏 → RR 减半 → HR 翻倍(实测 166bpm vs 真实80)。
#   静息/轻度活动 HR 不会超 133bpm, 故下限取 450ms 可硬性拦除次峰。
PPG_MAX_PEAK_MS    = 2000       # 最大峰间隔(ms) → 下限 30 bpm
PPG_PEAK_RATIO     = 0.5        # 峰值需超过近期幅度包络该比例(自适应门限系数)

# ==================== 上报参数 ====================
REPORT_INTERVAL_MS = 1000       # 串口上报间隔(ms)

# 当前路径为 MAX + MPU + ADS/GSR，雷达常量仅兼容历史诊断。
BLE_ENABLED = True
BLE_NAME = 'HeartCompanion'
USB_REPORT_ENABLED = True
GSR_READ_HZ = 20
PPG_RECOVERY_MS = 2000
PPG_MAX_AGE_MS = 150
PPG_CLOCK_DRIFT_MS = 200
PPG_MIN_QUALITY = 0.5
IMU_MAX_AGE_MS = 35
MPU6050_ALLOWED_IDS = (0x68, 0x70, 0x71, 0x73, 0x75, 0x98, 0x72)
USB_TX_BUFFER = 4096
