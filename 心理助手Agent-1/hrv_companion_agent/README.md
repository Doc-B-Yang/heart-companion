# 心理助手Agent：当前多通道版本

更新日期：2026-10-04。硬件为ESP32 + MAX30102 + MPU + ADS1115/GSR，雷达已移除。

联合采集 → USB/BLE原始schema=2 → 连续窗口/质量检查 → 本次佩戴确认和静息基线 → 13维特征 → 五类概率模型/输入域检查/10秒确认 → 分类或“不确定” → Agent陪伴和受保护的触发检查。

五类为平静、低焦虑、高焦虑、愉悦、疲劳。生理信号不能作为情绪真值；需用当前硬件的独立用户自评训练。旧HRV-only、MAD和运行时自动标签模型仅用于显式演示。没有合格新模型时，系统正常采集和陪伴，但输出“不确定”，不给虚构风险、PAD或主动生理干预。

## 运行

```powershell
Set-Location 'D:\大创心理助手\心理助手Agent-1\hrv_companion_agent'
python run_api.py
```

另一终端运行硬件开发/esp32_firmware/gateway.py --port COM6 --user participant01。打开 http://127.0.0.1:8000 ，网页编号与网关一致。Chrome/Edge也可直接BLE接收。

佩戴确认后静息约3分钟；整个约60秒基线窗口必须在电极120秒稳定期之后。重新佩戴、重启或数据超过5秒未到达，应重新确认校准。不会从情绪预测自动学习基线，也不会把噪声归一化成合格信号。

## 标注与训练

感受稳定约1分钟后选择自评。标签对应新鲜真实硬件窗口，必须与基线及其他标签不重叠。实验独立自评时应隐藏预测，避免模型提示影响标签。标签文件为data/multimodal_labels.jsonl；SQLite普通聊天自评不能代替完整通道的训练窗口。

```powershell
python models/train_multimodal.py --data data/multimodal_labels.jsonl --output models/multimodal_emotion.joblib
```

训练器检查标签来源、窗口、基线和相对特征一致性。按受试者留出测试，在训练受试者内分组校准概率；不随机拆相邻窗口，不以测试受试者拟合或校准，评估后不重新拟合全体数据。

至少8名受试者、每类10条独立窗口才能生成探索模型。生产启用条件：至少20名受试者，其中4名独立测试者，测试每类≥10窗口；balanced accuracy≥0.85、macro F1≥0.80、最低类别召回≥0.70、拒判后覆盖≥0.50、接受样本准确率≥0.85。条件是工程筛查，不是医学验证；不合格模型会保存探索报告，但加载器拒绝启用。

达标并完成真实独立复核后配置.env，重启后端：

```dotenv
MULTIMODAL_MODEL_ENABLED=1
MULTIMODAL_MODEL_PATH=models/multimodal_emotion.joblib
EMOTION_CONFIRM_SEC=10
```

相对模型路径按本包目录解析，只加载可信本机joblib文件。

## 接口

| 接口 | 用途 |
|---|---|
| POST /api/sensor/raw | 原始schema=2；拒绝模拟/历史回放；重复包不重跑 |
| POST /api/sensor/calibration | action=confirm_contact / capture_baseline / reset |
| POST /api/dataset/label | 独立自评，检查校准、新鲜度和不重叠 |
| GET /api/emotion/state、quality、affect | 共享过期检查与拒判 |
| /api/trigger/check | 同一硬件包仅检查一次；质量、校准和模型通过才可触发 |
| POST /api/chat/message | 没有可靠分类时以用户文字为依据 |
| GET /api/memory/export、POST /api/memory/purge | 包括新增多通道标签 |

旧/sensor/data缺少采集证据，不用于生产分类。手动情绪与模拟器标记simulation，不可保存训练标签。风险是工程触发评分；硬件分类不再换算成虚假的连续量表或PAD分数。

## 验证

```powershell
python run_tests.py
```

测试使用临时数据库和人工夹具，不访问LLM、不改用户记忆，不用于情绪准确率结论。硬件依赖见requirements-hardware.txt；此电脑入口也支持已有硬件开发/.runtime。

完整变更与真机待验事项见硬件开发/当前硬件与情绪分类改造说明-2026-10-04.md。说明书.md旧章节保留v2演示、工具和记忆说明；当前硬件入口、训练口径以本README为准。公网网页发布与本地实时连接见上级目录的《网页发布与实时连接说明.md》。
