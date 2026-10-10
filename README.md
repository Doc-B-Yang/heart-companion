# 心绪 · 心理助手实验项目

ESP32 + MAX30102 + GSR/ADS1115 + MPU 的生理信号采集、波形观察、自评标注与情感陪伴研究原型。

## 项目结构

- `心理助手Agent-1/`：Python 分析后端、网页、算法、测试与学习指南。
- `硬件开发/`：ESP32 固件、采集网关、真实波形服务、接线及实验说明。
- `.github/workflows/pages.yml`：仅发布静态网页的 GitHub Pages 工作流。

真实个人记录、聊天、密钥、环境依赖、文献全文与备份不上传仓库。网页没有模拟波形；断开设备时会显示未连接或历史数据。

## 在线网页

仓库 Settings → Pages → Source 选择 **GitHub Actions**。向 main 推送后工作流自动发布；网址见 Actions 的部署结果。Pages 只托管前端，不运行 Python，不保存生理记录，不包含大模型密钥。

## 本机分析与陪伴

```powershell
cd 心理助手Agent-1/hrv_companion_agent
python -m pip install -r requirements.txt -r requirements-hardware.txt
python run_api.py
```

访问 http://127.0.0.1:8000/。如需大模型回复，复制 `.env.example` 为 `.env` 并在本机填写密钥。

## 在线网页接实时设备

使用桌面 Chrome/Edge，在与 ESP32 相连的电脑上打开在线网址。详细操作见 `心理助手Agent-1/网页发布与实时连接说明.md`。浏览器对 HTTPS 网页访问本地服务有版本和权限限制；若无法连接，使用本机网页获得完整功能。

完整 PPG 波形由现有 USB 实验采集服务提供；常规 BLE 包只含心搏及皮肤电等摘要，不含完整 PPG 波形。波形采集服务与 USB 网关不能同时占用同一串口。

## 研究范围

PPG 的逐拍变异性称 PRV；GSR 目前记录电压，未标定为 µS。分类需要独立自评标签与留出验证；当前无已验证模型时保留“不确定”，不把波形好看当成情绪分类有效。

学习步骤见 `心理助手Agent-1/全流程学习与实验指南.md`。
