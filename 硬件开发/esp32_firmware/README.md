# ESP32 当前固件：MAX30102 + MPU + ADS1115/GSR

更新日期：2026-10-04。平台为现有ESP32-WROOM + CH340、MicroPython。雷达已移除；ld2410d.py仅为历史参考，不上传、不初始化、不参与判断。

## 接线和采集

全部模块使用3.3V并共地。SDA=GPIO21，SCL=GPIO22，当前I²C=400kHz，部署后以实测错误计数验证。

| 模块 | 地址/输入 | 用途 |
|---|---|---|
| MAX30102 | 0x57 | 100Hz IR PPG，FIFO平均=1，脉搏间期及质量诊断 |
| MPU6050/已记录兼容模块 | 0x68，可选INT=GPIO17 | 加速度、角速度、倾角运动门控；失败视为运动未知 |
| ADS1115 | 0x48，ADDR接GND，GSR输出→A0 | ±4.096V、128SPS单次转换，非阻塞轮询，约20读数/秒 |
| GSR | 3.3V，ADC→ADS A0 | 原始电压；未取得模块传递函数时不换算电导或SCR |

上电标定期间保持MPU静止。非标准ID=0x75仅表示本模块有限功能可测试，不能确认正品身份或全部兼容性。

## 程序

| 文件 | 运行端 | 职责 |
|---|---|---|
| main.py / acquisition.py | ESP32 | 唯一生产入口，三通道采集、健康状态、连续片段 |
| max30102.py / mpu6050.py / ads1115.py | ESP32 | 驱动及配置/读数检查 |
| hrv.py | ESP32 | 滤波、峰位、增量间期事件，异常或运动即断开片段 |
| serial_transport.py | ESP32 | 保留原有REPL输出，每次最多32字节，单条待发报告上限4096字节，不重配置UART0 |
| ble_transport.py | ESP32 | HC分片JSON、默认MTU23、协商后最高MTU247、有界队列 |
| gateway.py / ble_receiver.py | PC | USB/BLE统一送到后端/api/sensor/raw |
| host_monitor.py | PC | 诊断与CSV |
| selftest.py | ESP32 | 三器件基本连接和ADC转换检查 |
| deploy_firmware.py | PC | 备份、暂存、编译和回读验证，main最后替换 |
| firmware_sim.py / pipeline_check.py | PC | 显式旧HRV模拟、自洽性检查，不能证明情绪准确率 |

## 部署与运行

先关闭占用同一COM口的监视器或网关；端口以实际设备为准。

```powershell
Set-Location 'D:\大创心理助手\workbuddy\esp32_firmware'
python deploy_firmware.py --port COM6
```

部署允许列表包含新增serial_transport.py；雷达不在列表。2026-10-05已实际备份、部署并回读验证生产文件，USB与BLE均取得真实schema=2数据。恢复使用实际板端备份目录：

```powershell
python deploy_firmware.py --port COM6 --restore '实际板端备份目录'
```

终端1运行Agent目录的python run_api.py；终端2：

```powershell
Set-Location 'D:\大创心理助手\workbuddy\esp32_firmware'
python gateway.py --port COM6 --user participant01
```

打开 http://127.0.0.1:8000 ，网页编号改为participant01。确认佩戴，静息约3分钟，保存基线，再继续采集。标签窗口不能与基线或上条标签重叠。

BLE可在Chrome/Edge网页直接连接，或用网关：

```powershell
python gateway.py --ble --scan
python gateway.py --ble --address '扫描地址' --user participant01
```

## 数据协议和边界

原始包schema=2；模型特征版本独立为ppg-gsr-imu-baseline-voltage-v2。device_id/boot_id标识会话，seq递增，device_ms为开机时间，segment_id在连续性断开时改变。ibi_events=[心搏ID,结束毫秒,间期毫秒]；eda_samples=[采集毫秒,电压V]。生产后端不以最近三拍替代空增量。

BLE为适应Windows MTU23吞吐使用HC紧凑JSON v1（hc=1、v为固定顺序值数组），EDA时间戳相对device_ms传输；Python/网页重组后还原schema=2必需字段和绝对设备毫秒，再送同一API。USB保留完整诊断字段。紧凑格式不改变样本、质量阈值或去重身份；最高MTU247仅为兼容能力，实机Windows仍可能保持23。

错误字段ppg/imu/ads/fifo/clock，另有累计错误、采样率、在线状态、运动已知、通信丢弃。固件不输出mood、不诊断情绪。设备毫秒不作为Unix时间写入时间线。

当前约60秒窗口、PI、间期抖动及EDA噪声等门槛都是工程筛查。PPG是PRV代理，不能等同ECG金标准HRV。没有合格模型、未校准、信号错误、域外或置信不足均拒判。初始化失败需排查并重启，不在实时循环执行阻塞的重新标定。

2026-10-05短试已验证USB约100Hz采集、BLE紧凑格式35包无缺口和API去重。实际佩戴GC轮203个USB包均进入API，但缺28个序号、最大连续间期11，未获得合格基线。用户已独立自评平静；信号门控未通过，未保存基线或训练标签。峰值缓存现仅保留插值所需3点，与旧200点对照一致；372项测试通过。开发板上该内存优化的部署与复测结果见code-review-2026-10-04/实机验收-2026-10-05.md。GSR接触与参考信号验证仍未完成。完整对接见上级硬件到Agent对接说明.md。

峰值缓存优化已备份、部署并回读验证。非佩戴双通路检查USB连续230包、BLE连续173包均无序号缺口；除启动首包PPG/FIFO错误外，后续PPG/IMU/ADS/FIFO/clock均0，日志无未捕获内存异常。再次确认佩戴并重启标定后，四分钟USB收到231个连续包且API全部接受；PPG4/FIFO4错误在首包，IMU1在seq122，ADS/clock为0。PI中位数0.251%、最大0.379%，sig_quality最大0.49，全部窗口质量失败。未保存基线或训练标签，输出仍为“不确定”；该短段传输稳定性改善不能代替生理精度验证。

诊断采集可加 `--seconds 60 --record diagnostics/capture.jsonl`。记录中的 `phase=received` 是接收时保存的真实原始包，即使后端故障也保留；`phase=posted` 单独保存API结果和发送完成时刻。两种记录不能混算成两条传感器包。
