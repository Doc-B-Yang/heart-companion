# 心理助手Agent：个人实验与多通道采集

更新日期：2026-10-11。硬件为ESP32 + MAX30102 + MPU + ADS1115/GSR，雷达已移除。近期主线为真实双波形、独立自评、个人跨日高/低唤醒实验和带来源的陪伴交互；五分类与PAD列为条件性扩展。

真实采集 → USB/BLE schema=2 → 连续窗口/质量检查 → 本次佩戴确认和静息基线 → 特征与独立自评配对 → 离线个人唤醒实验。个人模型采用 `StandardScaler → LogisticRegression`，PPG/GSR共12项特征，尚未自动接入网页推理；1～9唤醒度评分中1～4为低、6～9为高、5暂不参与二分类。完整范围与判准见[项目开发计划书](../../项目开发计划书.md)。

五类为平静、低焦虑、高焦虑、愉悦、疲劳。生理信号不能作为情绪真值；需用当前硬件的独立用户自评训练。旧HRV-only、MAD和运行时自动标签模型仅用于显式演示。没有合格新模型时，系统正常采集和陪伴，但输出“不确定”，不给虚构风险、PAD或主动生理干预。

## 运行

此电脑推荐双击本目录的 `启动实验服务.cmd`，启动API（8000）和被动USB波形服务（8011）。不要让串口监视器或其他网关同时占用USB串口。只启动API可运行：

```powershell
Set-Location 'D:\大创心理助手\Heart-Companion\心理助手Agent-1\hrv_companion_agent'
python run_api.py
```

打开[本机网页](http://127.0.0.1:8000/)或[固定网址](https://doc-b-yang.github.io/heart-companion/)。Chrome/Edge可直接BLE接收，USB需要本机波形服务；分析、校准、自评存储与陪伴仍需本机API。GitHub Pages只托管静态前端，新功能需要部署更新后的网页才能在固定网址使用。操作见[Chrome实验使用说明](Chrome实验使用说明.md)。

佩戴确认后静息约3分钟；整个约60秒基线窗口必须在电极120秒稳定期之后。重新佩戴、重启或数据超过5秒未到达，应重新确认校准。不会从情绪预测自动学习基线，也不会把噪声归一化成合格信号。

## 自评、语言上下文与语音输入

“保存自评”可独立保存感受名称、文字及1～9唤醒度/愉悦度评分，无需合格生理信号；只有合格、不重叠的真实窗口才额外关联训练样本。后端离线时保留输入，恢复后重试。本人评分对应刚过去60秒；独立实验先评分、保存，再查看AI分析或模型解释。

新增 `affect_context` 与生理分类/PAD分开：评分按 `(score - 5) / 4` 归一化到[-1,1]，本人自评优先，保留来源、时间和5分钟期限。缺失轴保持未知，新一轮未填写的评分不继承旧评分；两轴都有有效依据才画二维点。自评可持久保存，展示上下文只在当前后端会话中短期使用。

点击“分析这段感受”或发送聊天后，在线AI可辅助估计本条表达中本人当前感受的**效价**，不推断唤醒度、支配度或医学诊断。过去、他人、引用、含糊表达应拒判；低置信、无效响应、离线或失败保持未知。自评与估计差异较大时保留冲突，不覆盖评分。置信门槛不是准确率；语言估计未经有效性验证，永不自动成为训练标签。

AI分析需配置可用的OpenAI兼容服务。按 `.env.example` 配置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_ID` 后重启后端；密钥仅在本机 `.env` 保存，不写入网页或公开仓库。本人评分无需AI在线，保存仍需API可用。

Chrome Web Speech API（`zh-CN`）在聊天和感受文字提供“语音输入”：由用户启动，转写写入输入框供修改，再手动发送、分析或保存；取消和失败保留原文。项目不录制或保存原始音频，但浏览器语音服务可能在线处理音频，文字分析会调用所配置AI服务。这里是转写后的语义分析，未实现声调/韵律情绪识别；可继续打字。

## 标注与训练（五分类扩展）

以下保留五分类训练/发布路线，不代表近期个人模型已接入网页。个人离线实验入口为 `models/experimental_arousal.py`；五分类另采用13维特征、160棵树随机森林和sigmoid概率校准。发布门槛是工程筛查，不是已取得成绩，也不是近期个人二分类判准。

感受稳定约1分钟后选择自评。标签对应新鲜真实硬件窗口，必须与基线及其他标签不重叠。实验独立自评时应隐藏预测，避免模型提示影响标签。标签文件为data/multimodal_labels.jsonl；SQLite普通聊天自评不能代替完整通道的训练窗口。

```powershell
python models/train_multimodal.py --data data/multimodal_labels.jsonl --output models/multimodal_emotion.joblib
```

训练器检查标签来源、窗口、基线和相对特征一致性。按受试者留出测试，在训练受试者内分组校准概率；不随机拆相邻窗口，不以测试受试者拟合或校准，评估后不重新拟合全体数据。

至少8名受试者、每类10条独立窗口才能生成探索模型。当前发布筛查要求：至少20名受试者，其中4名独立测试者，测试每类≥10窗口；balanced accuracy≥0.85、macro F1≥0.80、最低类别召回≥0.70、拒判后覆盖≥0.50、接受样本准确率≥0.85。条件是工程筛查，不是医学验证；不合格模型会保存探索报告，但加载器拒绝启用。

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
| POST /api/dataset/self-report | 独立保存本人自评，尝试关联合格窗口，返回是否形成训练样本及 `affect_context` |
| POST /api/dataset/label | 独立自评，检查校准、新鲜度和不重叠 |
| GET /api/emotion/state | 生理状态及独立 `affect_context`，包含过期检查 |
| POST /api/emotion/valence | `{user_id, text, input_source}`；只分析语言效价，不保存训练标签 |
| GET /api/emotion/quality、GET /api/emotion/affect | 原有生理质量与动态状态路径，不能代替主观上下文 |
| /api/trigger/check | 同一硬件包仅检查一次；质量、校准和模型通过才可触发 |
| POST /api/chat/message | `{user_id, message, input_source}`；以用户表达为依据，返回主观上下文 |
| GET /api/memory/export、POST /api/memory/purge | 包含多通道标签；清理同步移除当前主观上下文 |

`input_source` 可为 `text` 或 `speech_transcript`。旧/sensor/data缺少采集证据，不用于真实多通道分类。手动情绪与模拟器标记simulation，不可保存训练标签。风险是工程触发评分；硬件分类不再换算成虚假的连续量表或PAD分数。

## 验证

```powershell
python run_tests.py
```

软件测试使用临时数据库、人工信号或受控AI响应，不访问真实在线LLM、不改用户记忆，不证明情绪准确率。Chrome麦克风权限、在线AI、公网部署和真实设备分别验收；主观上下文可在无需佩戴的条件下先验证。硬件依赖见requirements-hardware.txt；此电脑入口也支持已有硬件运行时。

完整范围和待验事项以[项目开发计划书](../../项目开发计划书.md)为准。说明书.md旧章节保留v2演示、工具和记忆说明；当前硬件及新增上下文入口以本README为准。公网网页与本机连接见[网页发布与实时连接说明](../网页发布与实时连接说明.md)。
