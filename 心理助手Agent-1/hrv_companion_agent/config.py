# -*- coding: utf-8 -*-
"""
hrv_companion_agent 全局配置
============================
「基于实时心率变异性驱动的主动式情感陪伴系统」—— Agent 子系统配置

配置优先级: 代码默认值 < .env 环境变量 < 运行时显式传参
所有阈值均对应《项目开发计划书》§4.2 / 开题答辩 PPT STEP 04。
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径 ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = DATA_DIR / "logs"

for _d in (DATA_DIR, MODELS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _load_dotenv_files():
    """按优先级加载 .env：项目目录 → 当前目录 → 上级目录（不覆盖已加载值）"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidates = [PROJECT_ROOT / ".env", Path.cwd() / ".env"]
    try:
        candidates += [Path.cwd().parent / ".env",
                       Path.cwd().parent.parent / ".env",
                       PROJECT_ROOT.parent / ".env"]
    except Exception:  # noqa: BLE001
        pass
    seen = set()
    for p in candidates:
        if p.exists() and str(p.resolve()) not in seen:
            seen.add(str(p.resolve()))
            load_dotenv(p, override=False)


_load_dotenv_files()

# Load these AFTER dotenv, just like the other runtime options.
MULTIMODAL_MODEL_ENABLED = os.getenv('MULTIMODAL_MODEL_ENABLED', '0').lower() in ('1', 'true', 'yes')
MULTIMODAL_MODEL_PATH = os.getenv('MULTIMODAL_MODEL_PATH', str(MODELS_DIR / 'multimodal_emotion.joblib'))
if not Path(MULTIMODAL_MODEL_PATH).is_absolute():
    MULTIMODAL_MODEL_PATH = str(PROJECT_ROOT / MULTIMODAL_MODEL_PATH)
EMOTION_CONFIRM_SEC = max(3.0, float(os.getenv('EMOTION_CONFIRM_SEC', '10')))
SENSOR_STALE_SEC = 5.0
GSR_SETTLE_SEC = 120.0
ALLOW_SIMULATION = True


# ---------------------------------------------------------------- LLM ----
# API Key 兼容多种常见变量名（DeepSeek/OpenAI/通义千问/Kimi/智谱……）
def _key_source() -> str:
    for key in ("LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
                "DASHSCOPE_API_KEY", "QWEN_API_KEY", "MOONSHOT_API_KEY",
                "KIMI_API_KEY", "ZHIPU_API_KEY", "API_KEY"):
        if _env(key):
            return key
    return ""


def _infer_model(key_source: str) -> str:
    """根据命中的 Key 变量名推断默认模型名"""
    table = {
        "DEEPSEEK_API_KEY": "deepseek-chat",
        "OPENAI_API_KEY": "gpt-4o-mini",
        "DASHSCOPE_API_KEY": "qwen-plus",
        "QWEN_API_KEY": "qwen-plus",
        "MOONSHOT_API_KEY": "moonshot-v1-8k",
        "KIMI_API_KEY": "moonshot-v1-8k",
        "ZHIPU_API_KEY": "glm-4-flash",
    }
    return table.get(key_source, "deepseek-chat")


_KEY_SRC = _key_source()
LLM_MODEL_ID = _env("LLM_MODEL_ID") or _infer_model(_KEY_SRC)
LLM_API_KEY = _env("LLM_API_KEY") or _env("DEEPSEEK_API_KEY") \
    or _env("OPENAI_API_KEY") or _env("DASHSCOPE_API_KEY") \
    or _env("QWEN_API_KEY") or _env("MOONSHOT_API_KEY") \
    or _env("KIMI_API_KEY") or _env("ZHIPU_API_KEY") or _env("API_KEY")
LLM_BASE_URL = _env("LLM_BASE_URL", "https://api.deepseek.com")
LLM_PROVIDER = _env("LLM_PROVIDER", "auto")
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "1024"))
LLM_TIMEOUT = int(_env("LLM_TIMEOUT", "60"))

# --------------------------------------------------------------- TTS ----
TTS_ENABLED = _env("TTS_ENABLED", "0").lower() in ("1", "true", "yes")
TTS_VOICE = _env("TTS_VOICE", "zh-CN-XiaoyiNeural")  # 温暖女声
TTS_OUTPUT_DIR = DATA_DIR / "tts"

# ------------------------------------------------------------ 算法层 ----
# ML 分类器（MAD 数据集训练的随机森林），不存在时自动跳过 Layer 3
ML_MODEL_PATH = _env("ML_MODEL_PATH", str(MODELS_DIR / "emotion_model_rf.joblib"))
ML_CONFIDENCE_FLOOR = 0.6       # Layer3 激活条件: 前两层置信度低于该值
RULE_CONFIDENCE_FLOOR = 0.7     # Layer1 直接输出阈值
FUZZY_CONFIDENCE_FLOOR = 0.6    # Layer2 直接输出阈值

# HRV 计算窗口（秒）：与 ESP32 固件 5 分钟窗口保持一致
HRV_WINDOW_SEC = 300
IBI_MIN_MS = 300                # 伪迹过滤下界
IBI_MAX_MS = 2000               # 伪迹过滤上界

# ------------------------------------------------- 信号质量门禁（v2） ----
# 依据实测结论：PPG 存在「HR 正常但 HRV 是假的」这一隐蔽失效模式
# （HR 只看峰间距均值，抖动被平均掉；HRV 看逐拍差分，抖动全额进入）。
# 抖动指数 jitter = RMSSD / mean(IBI)，与 ESP32 固件 hrv.rr_jitter() 同定义。
QUALITY_GATE_ENABLED = _env("QUALITY_GATE_ENABLED", "1").lower() in ("1", "true", "yes")
QUALITY_JITTER_GOOD = 0.08      # ≤ 合格
QUALITY_JITTER_POOR = 0.12      # > 不可用（低质量数据不应参与情绪判定）
QUALITY_ECTOPIC_GOOD = 0.10     # 合格档允许的伪迹剔除比例
QUALITY_ECTOPIC_MAX = 0.20      # 伪迹剔除比例上限（超过即判不可用）
# 基线采集的质量门槛：脏基线会污染后续所有相对化特征
BASELINE_MIN_QUALITY = 0.5

# ------------------------------------------------------- PAD 维度模型 ----
# PADRegressor 需要先用 models/train_pad_model.py 训练出模型文件才生效；
# 未训练时自动走 RulePAD 规则映射（零依赖，始终可用）。
PAD_MODEL_PATH = _env("PAD_MODEL_PATH", str(MODELS_DIR / "pad_svr.joblib"))
PAD_ENABLED = _env("PAD_ENABLED", "1").lower() in ("1", "true", "yes")
# PAD 置信度低于该值时，不采用其 emotion_hint 去覆盖三层识别结果
# （文献2 实测 D 轴 ρc 仅 0.30，P/A 轴才可用）
PAD_MIN_CONFIDENCE = 0.35

# ---------------------------------------------------- 情感动力学（PWE）----
# 文献3 的 Plutchik 情绪轮动力学：给识别结果加上"连续性/累积/衰减"，
# 解决离散分类在陪伴场景中的情感跳变问题。
DYNAMICS_ENABLED = _env("DYNAMICS_ENABLED", "1").lower() in ("1", "true", "yes")
#: 用户气质档位：ideal(理想型) / indifferent(冷漠型) / sensitive(敏感型)
#:                / persistent(执着型) —— 取自文献3 表2
DYNAMICS_PERSONALITY = _env("DYNAMICS_PERSONALITY", "ideal")

# ------------------------------------------------------------ 触发引擎 ----
TRIGGER_LOW = 60        # R>60  → 低级干预（轻微问候）
TRIGGER_MEDIUM = 80     # R>80  → 主动对话
TRIGGER_HIGH = 90       # R>90  → 紧急关怀 + 呼吸练习
TRIGGER_EMERGENCY = 99  # R≥99 → 危机干预协议（转介专业求助，仅极端情形）
TRIGGER_COOLDOWN_SEC = 300      # 两次主动触发最小间隔（秒）
QUIET_CHECKIN_SEC = 2 * 3600    # 长时间无交互(默认2小时) → 低频主动问候
RISK_TREND_WINDOW = 10          # 风险趋势计算窗口（样本数）
BASELINE_ADAPT_RATE = 0.05      # 个体基线 EMA 更新速率
#: 心境（mood）对触发风险的放大系数 —— 让"今天整体状态差"的日子更容易被关怀。
#: 依据文献3 的心境累积思路：瞬时风险之外，慢变量也应参与决策。
TRIGGER_MOOD_WEIGHT = 12.0
#: 情感动力学强度对风险的贡献上限（避免动力学完全主导阈值语义）
TRIGGER_INTENSITY_WEIGHT = 15.0
#: 触发需连续满足的最小次数（抗单点误报）。
#: 默认 1 = 关闭该保护，保持与 v1 的触发语义一致；
#: 长时间监护场景建议设为 2~3（配合 1 包/秒的上报，代价是延迟 1~2 秒）。
TRIGGER_MIN_CONSECUTIVE = int(_env("TRIGGER_MIN_CONSECUTIVE", "1"))

# ------------------------------------------------------- 语义区间 ----
# 把连续数值折成"人话"的**唯一定义处**。
#
# ⚠️ 为什么必须集中定义：这些区间同时被两处消费 ——
#    · backend/views.py   → 给 APP 显示（用短标签）
#    · agent/prompts.py   → 给模型提示（用短标签 + 表达提示）
# 各自维护一份的后果不是报错，而是**跨层矛盾**：前端显示"偏高"、
# 模型被告知"中等"。两边单看都没错，只有放在一起才看得出来。
#
#: 风险评分 0~100 → (下界, 短标签, 给模型的表达提示)
RISK_BANDS = (
    (80, "很高", "明显紧张/不适，需要安抚"),
    (60, "偏高", "有些紧绷"),
    (40, "中等", "基本平稳，略有波动"),
    (20, "较低", "比较放松"),
    (0,  "低", "很放松"),
)

#: 信号质量等级 → 给模型的措辞。
#:
#: ⚠️ 键必须与 ``HRVExtractor.quality_metrics()["quality_level"]`` 的**实际
#: 产出值**完全一致（合格/勉强/不可用/不足）。曾用 good/fair/poor 作键，
#: 由于查不到任何键而**静默返回空串**，导致"信号质量: …"这一行从未被
#: 注入过 —— 不报错、不告警，只是模型永远不知道信号好不好。
QUALITY_LEVEL_LABELS = {
    "合格": "良好（生理判断可用）",
    "勉强": "一般（可作参考，结论以用户自述为准）",
    "不可用": "不佳（不可用于判断）",
    "不足": "数据积累中（尚不能判断）",
}

# ------------------------------------------------------------ 记忆层 ----
MEMORY_DB_PATH = Path(_env('MEMORY_DB_PATH', str(DATA_DIR / 'memory.db')))
MEMORY_TOP_K = 5                # 语义检索条数
RECENT_CONTEXT_N = 6            # 注入 Prompt 的最近互动条数
CHROMA_ENABLED = _env("CHROMA_ENABLED", "0").lower() in ("1", "true", "yes")

# ---- FTA 情境记忆单元（文献5 FTA-Mem: Fact-Time-Affect 锚定）----
# 记忆单元粒度取"情境级"——文献5 实测情境级 > 会话级 > 轮对级
# （在证据保留与构造成本之间折中最佳）。
MEMORY_FTA_ENABLED = _env("MEMORY_FTA_ENABLED", "1").lower() in ("1", "true", "yes")
# 检索混合打分 R = λ·向量相似度 + (1−λ)·结构化线索匹配（文献5 式(8)，λ=0.8）
MEMORY_RETRIEVAL_LAMBDA = 0.8
# 检索分数阈值 τ 与 Top-K（文献5 式(9)：τ=0.35, K=10）
MEMORY_RETRIEVAL_TAU = 0.35
MEMORY_FTA_TOP_K = 6

# ---- partial 单元融合阈值（文献5 式(4)(5) 的跨段携带）----
# 语义：用户一句话说了一半就走了（partial），下次补充时应当**融合进**
# 原单元，而不是新开一条 —— 否则同一件事会被记住两遍。
#
# ⚠️ 这个阈值只在**词面连续**时才会触发，实测边界如下：
#     · "和室友的关系有点" → "和室友关系有点紧张，因为作息不同"
#       → 重叠 ≈ 0.8，融合 ✓
#     · "我最近在准备一个考试" → "就是下周五的高数补考，还没复习"
#       → 重叠 = 0.0，**不融合** ✗（两句语义相关但无共同 bigram）
#   根因是无嵌入模型时的相似度代理只能看字符 bigram（见 memory_manager._similar
#   的量纲讨论）。要覆盖第二种情形，需要接真向量后端（text2vec/ChromaDB，
#   CHROMA_ENABLED=1）或引入 LLM 边界检测。
#   **在升级之前，不要对外宣称"系统能自动把同一件事的两次提及连起来"。**
# 取值 0.35 与检索阈值 τ 同量级，偏向"宁可不融合也不误融合" ——
# 对心理陪伴场景，把两件不相干的事合并成一条记忆比漏融合更糟。
MEMORY_FUSE_OVERLAP = 0.35
# 艾宾浩斯遗忘曲线（文献4 MemoryBank）：保留度 R = e^(−t/S)，
# 记忆被召回一次则 S += 1 并重置计时 —— "复述使遗忘曲线变平"。
MEMORY_FORGET_ENABLED = _env("MEMORY_FORGET_ENABLED", "1").lower() in ("1", "true", "yes")
MEMORY_STRENGTH_INIT = 1.0
MEMORY_STRENGTH_GAIN = 1.0      # 每次召回强度增量
MEMORY_RETENTION_FLOOR = 0.15   # 低于该保留度 → 归档（不删除，便于调试与合规）

# ------------------------------------------------------- 对话风格调制 ----
# 文献3 式(21)–(26)：把情感强度与心境映射到 LLM 的**外部**超参。
# ⚠️ 落地边界：本项目调用云端 API 模型，无法改动 Transformer 内部结构
#    （注意力头数/隐藏层维度/层数等，文献3 式(27)–(33)），故只调制外部参数。
LLM_PARAM_MODULATION = _env("LLM_PARAM_MODULATION", "1").lower() in ("1", "true", "yes")
LLM_TEMP_NEUTRAL = 0.7          # 中性温度（文献3 的 Tin）
LLM_TEMP_RANGE = (0.4, 1.1)     # 温度可用区间（防止调制越界）
LLM_TOPP_NEUTRAL = 0.9
LLM_TOPP_RANGE = (0.7, 1.0)

# ------------------------------------------------------- 安全与伦理 ----
# 专业边界：AI 是陪伴与自我调节辅助，不是医疗诊断/治疗。
SAFETY_ENABLED = _env("SAFETY_ENABLED", "1").lower() in ("1", "true", "yes")
# 危机风险分级（对应安全策略三档），与触发引擎的 high/emergency 联动
CRISIS_LEVELS = ("low", "medium", "high")
# 首次会话需展示的知情同意要点（长期记忆的启用必须告知用户）
PRIVACY_NOTICE_VERSION = "v1.1"
# 是否允许在对话中出现生理指标数值（默认禁止，改由自然语言表达）
EXPOSE_RAW_METRICS = _env("EXPOSE_RAW_METRICS", "0").lower() in ("1", "true", "yes")

# ------------------------------------------------------- 转介分层（v2）----
# 文献6（《心智和算法共舞》行业报道）指出：AI 心理助手的产品形态必须收敛到
# 「风险分级预警 + 人工专业联动」，即系统是**分层支持的第一层**，
# 出现超出陪伴范围的问题时必须把用户**转出去**，而不是留在系统里。
#
# 因此本系统的转介不是"一个热线列表"，而是一条四级阶梯：
#   L1 自助  → L2 校内/社会支持 → L3 临床专业 → L4 紧急处置
# 每一级给出：适用判据、建议动作、以及"如何开口"的话术脚本。
#
# ⚠️ 需按实际学校/城市填写：CAMPUS_RESOURCES 中的占位条目请替换为本校
#    心理健康教育中心的真实电话与地址（运行 run_cli.py 时会提示）。
REFERRAL_LADDER = [
    {
        "level": "L1",
        "key": "self_help",
        "name": "自助调节",
        "when": "轻中度情绪波动，生活功能未受损，能自我安抚与恢复。",
        "actions": [
            "使用系统内的呼吸引导、躯体安抚、认知解离等技能",
            "规律作息与运动（每周 ≥3 次、每次 ≥30 分钟有氧对情绪有稳定作用）",
            "记录情绪日记，识别触发点与有效应对方式",
        ],
    },
    {
        "level": "L2",
        "key": "social_support",
        "name": "身边支持网络",
        "when": "情绪困扰持续数天至两周，或与具体事件（考试/关系/家庭）相关。",
        "actions": [
            "先告诉一个可信任的人（室友/朋友/家人/辅导员/导师）",
            "联系学校心理健康教育中心做一次免费面谈（多数高校对学生免费且保密）",
            "参加团体辅导或心理讲座，降低“只有我这样”的孤立感",
        ],
    },
    {
        "level": "L3",
        "key": "clinical",
        "name": "临床专业评估",
        "when": "情绪低落/焦虑持续两周以上、明显影响睡眠·进食·学业·人际，"
                "或出现自评量表提示的中重度症状。",
        "actions": [
            "预约三甲医院精神科（心理科）门诊做正规评估 —— 评估不等于“有病”",
            "必要时由医生判断是否需要药物或心理治疗（本系统无权也无能力判断）",
            "如已在校心理中心咨询，可请咨询师协助转介",
        ],
    },
    {
        "level": "L4",
        "key": "emergency",
        "name": "紧急处置",
        "when": "出现自伤/自杀意念、计划或准备行为；或严重躯体症状。",
        "actions": [
            "立即确保身边有人陪护，移除可能造成伤害的物品",
            "拨打全国24小时心理援助热线 400-161-9995，或 110 / 120",
            "告知身边可信任的人并请求陪伴，不要独自度过",
        ],
    },
]

#: 校内与社会资源（**请按本校实际情况替换占位内容**）
CAMPUS_RESOURCES = [
    ("学校心理健康教育中心", "请填写本校预约电话/地址",
     "多数高校对本校学生免费、保密，可线上预约"),
    ("辅导员 / 班主任", "请填写联系方式", "可协助请假、转介与日常支持"),
    ("校医院", "请填写门诊电话", "躯体症状与睡眠问题的第一站"),
    ("三甲医院精神科门诊", "请填写目标医院", "正规评估与治疗，需预约挂号"),
]

#: 心理教育免责声明（心理教育≠诊断，需在相关回复中自然带出）
PSYCHOEDUCATION_DISCLAIMER = (
    "以上是通用的心理学知识，用于帮助理解自己的状态，"
    "**不构成诊断**。如果困扰持续或加重，请让专业人士陪你一起看。")

# ------------------------------------------------------------ 角色设定 ----
COMPANION_NAME = "小暖"
DEFAULT_USER_ID = "user001"

# 危机热线（国内权威心理援助资源，危机工具使用）
CRISIS_HOTLINES = [
    ("全国24小时心理援助热线", "400-161-9995"),
    ("北京心理危机研究与干预中心", "010-82951332"),
    ("教育部华中师大心理援助热线", "400-9678-920"),
    ("青少年心理咨询与法律援助热线", "12355"),
    ("紧急情况请拨打", "110 / 120"),
]

# ------------------------------------------------- 主观-客观校准（v2）----
# 大创核心指标「情绪识别准确率 > 85%」需要**真值**，而生理信号没有真值。
# 因此系统内置主观自评通道（1~10 分），用于：
#   1) 主客观不一致时，以用户自述为准（伦理要求：用户有最终解释权）
#   2) 累积个人标注数据，为后续 20 人实验提供真值来源
#   3) 校准个体基线 —— 缓解「腕/指式 PPG 灌注差异导致阈值漂移」
SELF_REPORT_ENABLED = _env("SELF_REPORT_ENABLED", "1").lower() in ("1", "true", "yes")
#: 主观与客观差异超过该阈值 → 判定为「不一致」，触发以用户为准的降权
SELF_REPORT_CONFLICT_GAP = 3.0
#: 不一致时生理侧置信度乘的折扣
SELF_REPORT_CONFLICT_DISCOUNT = 0.5


def as_dict() -> dict:
    """导出配置快照（供日志/说明使用），仅包含可序列化字段"""
    return {
        "llm": {
            "model": LLM_MODEL_ID,
            "base_url": LLM_BASE_URL,
            "provider": LLM_PROVIDER,
            "online": bool(LLM_API_KEY),
            "temperature": LLM_TEMPERATURE,
        },
        "tts": {"enabled": TTS_ENABLED, "voice": TTS_VOICE},
        "algorithm": {
            "ml_model_path": ML_MODEL_PATH,
            "hrv_window_sec": HRV_WINDOW_SEC,
            "quality_gate": QUALITY_GATE_ENABLED,
            "quality_jitter_poor": QUALITY_JITTER_POOR,
        },
        "pad": {
            "enabled": PAD_ENABLED,
            "model_path": PAD_MODEL_PATH,
            "min_confidence": PAD_MIN_CONFIDENCE,
        },
        "dynamics": {
            "enabled": DYNAMICS_ENABLED,
            "personality": DYNAMICS_PERSONALITY,
        },
        "trigger": {
            "low": TRIGGER_LOW,
            "medium": TRIGGER_MEDIUM,
            "high": TRIGGER_HIGH,
            "emergency": TRIGGER_EMERGENCY,
            "cooldown_sec": TRIGGER_COOLDOWN_SEC,
            "quiet_checkin_sec": QUIET_CHECKIN_SEC,
            "mood_weight": TRIGGER_MOOD_WEIGHT,
            "min_consecutive": TRIGGER_MIN_CONSECUTIVE,
        },
        "memory": {"db": str(MEMORY_DB_PATH), "chroma": CHROMA_ENABLED,
                   "fta_enabled": MEMORY_FTA_ENABLED,
                   "forget_curve": MEMORY_FORGET_ENABLED,
                   "retrieval_lambda": MEMORY_RETRIEVAL_LAMBDA},
        "llm_modulation": LLM_PARAM_MODULATION,
        "safety": {"enabled": SAFETY_ENABLED,
                   "privacy_notice": PRIVACY_NOTICE_VERSION,
                   "referral_levels": [x["key"] for x in REFERRAL_LADDER]},
        "self_report": {"enabled": SELF_REPORT_ENABLED,
                        "conflict_gap": SELF_REPORT_CONFLICT_GAP},
        "companion": {"name": COMPANION_NAME},
    }
