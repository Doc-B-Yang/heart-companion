# -*- coding: utf-8 -*-
"""
host_monitor.py - PC 端串口监听工具（运行在电脑上，非 ESP32 上）

功能:
  - 读取 ESP32 上报的 JSON 行
  - 输出 PPG / IMU / ADS / GSR 健康状态和诊断量
  - 可选导出 CSV

用法:
    python host_monitor.py --port COM6 --baud 115200
    python host_monitor.py --port COM6 --csv data_log.csv
"""

import argparse
import csv
import json
import sys
import time

try:
    import serial
except ImportError:
    print("缺少 pyserial，请先: pip install pyserial")
    sys.exit(1)

CSV_FIELDS = ['device_ms','seq','boot_id','hr','sdnn','rmssd','pnn50','nn',
              'sig_quality','continuity','moving','motion_known','eda_voltage','ads','errors']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM6')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--csv', default=None, help='导出 CSV 文件路径')
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print("已连接 %s @%d，等待数据 (Ctrl-C 退出)..." % (args.port, args.baud))

    fcsv = None
    writer = None
    if args.csv:
        fcsv = open(args.csv, 'w', newline='', encoding='utf-8')
        writer = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()

    try:
        while True:
            line = ser.readline()
            if not line:
                continue
            s = line.decode('utf-8', 'replace').strip()
            if not s:
                continue
            # 非 JSON 行(启动日志/调试) 直接打印
            if not s.startswith('{'):
                print("  ", s)
                continue
            try:
                d = json.loads(s)
            except ValueError:
                print("  [raw]", s)
                continue

            print(json.dumps({k:d.get(k) for k in CSV_FIELDS},ensure_ascii=False))

            if writer:
                row = {k:d.get(k) for k in CSV_FIELDS}
                writer.writerow(row)
                fcsv.flush()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        ser.close()
        if fcsv:
            fcsv.close()
            print("CSV 已保存:", args.csv)


if __name__ == '__main__':
    main()
