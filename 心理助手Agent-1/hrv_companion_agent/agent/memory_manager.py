# -*- coding: utf-8 -*-
"""
长期记忆管理器（v2 · FTA 情境记忆）
===================================
对应计划书 §4.3 memory_manager.py。

v1 的记忆是「对话流水 + 情绪时间线」两件事，够跑通闭环，但撑不起"长期陪伴"：
  * 一次困扰会被拆成十几条孤立对话记录，检索时召回的是**碎片**；
  * 记忆只增不减，越用越慢，"三周前随口一提的小事"和"昨天的崩溃"等权竞争；
  * 完全没有"某件事后来怎么样了"的追踪能力，所以问不出
    "上次你说要联系导师，后来怎么样了？"这种最有陪伴感的话。

v2 依据两篇文献重构记忆单元与检索，同时**完整保留 v1 全部接口**。

--------------------------------------------------------------------------------
文献4《MemoryBank: Enhancing LLMs with Long-Term Memory》(arXiv:2305.10250,
AAAI 2024) —— 贡献是**记忆的三个层次 + 遗忘曲线**：

  1. 细粒度对话（in-depth memory，带时间戳）
  2. 分层事件摘要（日摘要 → 全局摘要）
  3. 动态人格画像（日洞察 → 全局画像）
  记忆更新用艾宾浩斯遗忘曲线：**R = e^(−t/S)**
    * R = 保留度，t = 距上次学习的时间，S = 记忆强度
    * **每被召回一次 S += 1 且 t 重置为 0** —— "复述使遗忘曲线变平"
    ⚠️ 原文自称这是"高度简化、探索性"的工程类比，不是心理学定论，
       引用时必须注明。

文献5《FTA-Mem: Fact-Time-Affect Anchored Memory for Low-Density Long-Term
Dialogue》(arXiv:2608.16303v1) —— 贡献是**记忆单元的粒度与结构**：

  * 论点：长期情绪支持对话是**低密度**的（单轮不完整、证据稀疏、跨会话分布），
    所以"多检索一些"或"压缩更长的上下文"都救不了 —— **问题出在记忆单元的粒度**。
    实测粒度效果：**情境级 > 会话级 > 轮对级**
    （情境级 38.71 F1 / 66.68 BERTScore，且比轮对级**成本更低**：
      6564 单元 vs 8556 单元，3249 次调用 vs 4883 次）；
  * 单元定义 **五元组 m = ⟨x_F, x_T, x_A, e, o⟩**
    （事实锚 / 时间锚 / 情感锚 / 证据指针 / 构造状态）；
  * 检索用**混合打分**（式8）：R(q,m) = λ·s_emb + (1−λ)·c_structured，
    原文取 λ = 0.8，阈值 τ = 0.35，Top-K = 10；
  * **partial 单元跨段携带**（式4）：上一轮只说半句的事，先存为 partial，
    下一轮补全后**融合**（式5）再分配正式 ID，避免一件事被拆成两条无关记忆；
  * **时序链接**（式6）：新单元与历史单元建立
    `updates / contradicts / supports / followup_of` 关系 —— 这正是
    "某件事后来怎么样了"能问得出来的原因；
  * 消融实验：**去掉时间信息退化最大，去掉情感信息也显著退化**，
    去掉关系链接影响较小 —— 说明"时间 + 情感 + 事实"三锚缺一不可；
  * 伦理要求（原文明确）：长期记忆会引入**跨会话持续影响**的风险
    （错误/过时/过度解读的记忆会反复作用于后续回复），部署需
    **知情同意、查看/更正/删除/留存控制、访问控制、数据最小化**。
    → v2 因此实现了 ``export_user_data`` / ``purge_user_data`` 等数据权利接口。

本项目的**原创增量**（两篇文献都没有的）
--------------------------------------------------------------------------------
在 FTA 情感锚之外，额外存 **``physio_anchor``（生理锚）**：每次会话的
HRV 指标（SDNN/RMSSD/LF-HF/抖动指数等）随记忆一起落库，并与 ``emotion``
字段做**交叉校验**。这使"这段记忆形成时的生理状态"可回溯，
是 HRV 主线相对纯文本记忆系统的差异化贡献，可作为论文创新点。
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config

logger = logging.getLogger("companion.memory")

# ======================================================================
# FTA 记忆单元的类型体系（文献5 原文取值）
# ======================================================================
#: 单元内容类型（content_type）—— 文献5 的枚举
CONTENT_TYPES = ("event", "current_state", "relationship_event",
                 "belief_candidate", "homework_or_plan", "support_interaction",
                 "risk_signal", "general_fact")

#: 单元完成度（构造状态 o）—— 文献5 式(3)(4)
COMPLETENESS = ("complete", "partial", "pending", "unknown")

#: 生命周期状态 —— 文献5 的 lifecycle
LIFECYCLE = ("pending", "completed", "contradicted", "archived", "unknown")

#: 时间取向（时间锚的组成部分）
TIME_ORIENTATIONS = ("past", "present", "future", "mixed", "unknown")

#: 记忆间时序关系（文献5 式(6)）
LINK_RELATIONS = ("same_episode_as", "updates", "contradicts",
                  "followup_of", "supports", "unknown")

#: 边界触发类型（文献5 的 BWS 边界线索，用于情境切分）
BOUNDARY_TRIGGERS = ("topic_change", "goal_change", "emotion_shift",
                     "new_time_reference", "new_person_or_relationship",
                     "problem_to_support_transition",
                     "support_strategy_transition",
                     "homework_or_plan_creation",
                     "correction_or_contradiction", "risk_signal",
                     "session_start", "unknown")

#: 边界线索的中文触发词（无 LLM 时的规则版切分；有 LLM 时可升级为模型判定）
_BOUNDARY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "risk_signal": ("不想活", "自杀", "自残", "割腕", "活着没意思", "结束生命",
                    "死了算了", "跳楼", "安眠药", "活不下去"),
    "emotion_shift": ("突然", "忽然", "一下子", "崩溃", "忍不住", "泪崩",
                      "好难受", "撑不住", "喘不过气"),
    "homework_or_plan_creation": ("我决定", "我打算", "我要去", "计划",
                                  "明天就", "从今天开始", "我答应", "试试看"),
    "correction_or_contradiction": ("其实不是", "说错了", "更正", "不是这样",
                                    "我搞错了", "记错了"),
    "new_person_or_relationship": ("我妈", "我爸", "老师", "室友", "男朋友",
                                   "女朋友", "导师", "朋友", "同学"),
    "new_time_reference": ("上周", "昨天", "明天", "下周", "小时候", "去年",
                           "以后", "将来"),
    "topic_change": ("对了", "另外", "说到这个", "还有一件事", "换个话题"),
}

#: 情绪 → 情境类型的粗映射（供无 LLM 时兜底）
_EMOTION_TO_SITUATION = {
    "高焦虑": "current_state", "低焦虑": "current_state",
    "疲劳": "current_state", "愉悦": "event", "平静": "general_fact",
}


class MemoryManager:
    """SQLite 长期记忆（v1 时间线/画像 + v2 FTA 情境记忆 + 遗忘曲线）

    三层结构（对齐文献4 MemoryBank 与文献5 FTA-Mem）::

        turns / interactions     原始对话（细粒度，带时间戳）
        fta_units                情境级记忆单元（事实锚+时间锚+情感锚+生理锚）
        memory_links             单元间时序关系（updates/contradicts/followup_of…）
        daily_summary            日摘要（分层摘要的中层）
        user_portrait            动态人格画像（分层摘要的高层）
    """

    def __init__(self, user_id: str, db_path=None):
        self.user_id = user_id
        self.db_path = str(db_path or config.MEMORY_DB_PATH)
        # RLock（可重入）而非 Lock：本类存在"持锁方法内部再调用另一个
        # 持锁方法"的合理场景（如 export_user_data 汇总各表）。
        # 用普通 Lock 会在同线程二次 acquire 时**自死锁**且极难排查
        # （现象是调用永久挂起、无异常、无日志）。
        self._lock = threading.RLock()
        self._chroma = self._init_chroma()
        # 持久连接（:memory: 数据库必须共享同一连接，否则每次都是全新空库）
        self._db_conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        self._db_conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------ 初始化
    def _init_db(self):
        with self._lock, self._conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                profile TEXT NOT NULL DEFAULT '{}',
                baseline TEXT NOT NULL DEFAULT '{}',
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                ts REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion TEXT,
                risk REAL
            );
            CREATE TABLE IF NOT EXISTS emotion_timeline (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                ts REAL NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL,
                risk REAL,
                hr_bpm REAL,
                sdnn_ms REAL,
                rmssd_ms REAL,
                lf_hf REAL,
                layer TEXT
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                ts REAL NOT NULL,
                strategy TEXT,
                before_risk REAL,
                after_risk REAL,
                effective INTEGER
            );
            -- ================= v2: FTA 情境记忆单元 =================
            -- 五元组 ⟨事实锚 x_F, 时间锚 x_T, 情感锚 x_A, 证据指针 e, 构造状态 o⟩
            -- 另加本项目的原创字段 physio_json（生理锚）
            CREATE TABLE IF NOT EXISTS fta_units (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT NOT NULL,
                created_ts      REAL NOT NULL,
                fact_text       TEXT,
                situation_type  TEXT,
                participants    TEXT,
                dialogue_ts     REAL,
                event_time      TEXT,
                time_orientation TEXT,
                status          TEXT,
                emotion         TEXT,
                emotion_conf    REAL,
                intention       TEXT,
                relation_hint   TEXT,
                physio_json     TEXT,
                evidence_turn_id INTEGER,
                evidence_span   TEXT,
                completeness    TEXT DEFAULT 'complete',
                lifecycle       TEXT DEFAULT 'pending',
                strength        REAL DEFAULT 1.0,
                last_recall_ts  REAL,
                recall_count    INTEGER DEFAULT 0,
                archived        INTEGER DEFAULT 0
            );
            -- 时序关系图（文献5 式(6)）
            CREATE TABLE IF NOT EXISTS memory_links (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                src_id      INTEGER NOT NULL,
                dst_id      INTEGER NOT NULL,
                relation    TEXT NOT NULL,
                confidence  REAL DEFAULT 0.5,
                created_ts  REAL
            );
            -- 分层摘要（文献4 MemoryBank: 日摘要 → 全局摘要）
            CREATE TABLE IF NOT EXISTS daily_summary (
                user_id     TEXT NOT NULL,
                day         TEXT NOT NULL,
                summary     TEXT,
                emotion_stat TEXT,
                updated_ts  REAL,
                PRIMARY KEY (user_id, day)
            );
            -- 动态人格画像（文献4 MemoryBank: 日洞察 → 全局画像）
            CREATE TABLE IF NOT EXISTS user_portrait (
                user_id     TEXT PRIMARY KEY,
                portrait    TEXT,
                updated_ts  REAL
            );
            -- ========== v2: 主观自评（大创实验真值来源） ==========
            -- 生理信号没有真值：情绪识别准确率 >85% 这一指标必须有独立标注。
            -- 本表把用户的主观评分与其时的生理识别结果**成对**存下，用途有三：
            --   1) 伦理：主客观冲突时以用户为准（用户对自身感受有最终解释权）
            --   2) 实验：累积带真值的样本，供后续 20 人实验直接使用
            --   3) 校准：识别个体的"生理-主观"映射偏差，用于基线校正
            CREATE TABLE IF NOT EXISTS self_report (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                ts          REAL NOT NULL,
                score       REAL,            -- 主观不适程度 0~10
                emotion     TEXT,            -- 主观情绪标签（可选）
                note        TEXT,            -- 主观补充描述
                -- 同一时刻的客观侧快照（用于算一致性）
                o_emotion   TEXT,
                o_confidence REAL,
                o_risk      REAL,
                o_pad_p     REAL,
                o_pad_a     REAL,
                o_pad_d     REAL,
                o_jitter    REAL,
                o_quality   TEXT,
                conflict    INTEGER DEFAULT 0,  -- 主客观是否冲突
                gap         REAL                -- 冲突幅度
            );
            CREATE INDEX IF NOT EXISTS idx_int_user_ts ON interactions(user_id, ts);
            CREATE INDEX IF NOT EXISTS idx_tl_user_ts ON emotion_timeline(user_id, ts);
            CREATE INDEX IF NOT EXISTS idx_fta_user_ts ON fta_units(user_id, created_ts);
            CREATE INDEX IF NOT EXISTS idx_fta_complete ON fta_units(user_id, completeness);
            CREATE INDEX IF NOT EXISTS idx_link_src ON memory_links(user_id, src_id);
            CREATE INDEX IF NOT EXISTS idx_link_dst ON memory_links(user_id, dst_id);
            CREATE INDEX IF NOT EXISTS idx_sr_user_ts ON self_report(user_id, ts);
            """)
            self._ensure_columns(conn)
            conn.execute(
                "INSERT OR IGNORE INTO users(user_id, created_at) VALUES(?, ?)",
                (self.user_id, time.time()))
            conn.commit()

    @staticmethod
    def _ensure_columns(conn):
        """增量迁移：给旧库补 v2 新增列（幂等，可重复执行）

        现场已有 memory.db 时直接 ALTER，避免用户需要删库重建。
        """
        wanted = {
            "emotion_timeline": [
                ("jitter", "REAL"), ("quality_level", "TEXT"),
                ("pnn50", "REAL"), ("sd_hr_bpm", "REAL"),
                ("pad_p", "REAL"), ("pad_a", "REAL"), ("pad_d", "REAL"),
            ],
            "interactions": [("fta_id", "INTEGER")],
        }
        for table, cols in wanted.items():
            try:
                have = {r["name"] for r in
                        conn.execute(f"PRAGMA table_info({table})").fetchall()}
            except sqlite3.Error:
                continue
            for name, decl in cols:
                if name not in have:
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                    except sqlite3.Error as e:   # noqa: BLE001
                        logger.debug("列 %s.%s 添加跳过: %s", table, name, e)

    def _init_chroma(self):
        """可选向量库：ChromaDB 语义检索增强（未安装/未开启时静默禁用）"""
        if not config.CHROMA_ENABLED:
            return None
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=str(config.DATA_DIR / "chroma"))
            return client.get_or_create_collection(f"memory_{self.user_id}")
        except Exception as e:  # noqa: BLE001
            logger.info("ChromaDB 未启用: %s → 使用混合关键词检索", e)
            return None

    def _conn(self) -> sqlite3.Connection:
        return self._db_conn

    # ==================================================================
    # 用户画像 / 生理基线（v1 接口保留）
    # ==================================================================
    def get_user_profile(self) -> dict:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT profile FROM users WHERE user_id=?",
                               (self.user_id,)).fetchone()
        return json.loads(row["profile"]) if row else {}

    def update_user_profile(self, updates: dict, merge: bool = True):
        """更新画像（merge=True 时浅合并）"""
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT profile FROM users WHERE user_id=?",
                               (self.user_id,)).fetchone()
            profile = json.loads(row["profile"]) if row else {}
            if merge:
                profile.update(updates)
            else:
                profile = updates
            conn.execute("UPDATE users SET profile=? WHERE user_id=?",
                         (json.dumps(profile, ensure_ascii=False), self.user_id))
            conn.commit()
        if self._chroma:
            try:
                self._chroma.upsert(
                    ids=["profile"],
                    documents=[json.dumps(profile, ensure_ascii=False)],
                    metadatas=[{"kind": "profile"}])
            except Exception:  # noqa: BLE001
                pass

    def get_baseline(self) -> dict:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT baseline FROM users WHERE user_id=?",
                               (self.user_id,)).fetchone()
        return json.loads(row["baseline"]) if row else {}

    def update_baseline(self, feats: dict, quality: Optional[dict] = None):
        """EMA 平滑更新个体基线（CH.02 个体差异挑战的应对）

        v2: 增加质量门槛 —— **脏基线会污染后续所有相对化特征**
        （文献1 的相对特征值法把基线当作分母），因此只在信号质量达标时更新。
        这与 ESP32 侧 `bridge.py --baseline` 的要求一致。
        """
        if config.QUALITY_GATE_ENABLED and quality is not None:
            if not quality.get("quality_ok", True):
                logger.info("基线更新跳过：信号质量不达标（%s）",
                            quality.get("quality_level"))
                return False
        alpha = config.BASELINE_ADAPT_RATE
        # v2: 相对化用到的特征都要进基线，否则相对特征值法只能覆盖一半
        keys = ("hr_bpm", "sdnn_ms", "rmssd_ms", "lf_hf", "pnn50_pct",
                "sdsd_ms", "sd1_ms", "sd2_ms", "sd_hr_bpm",
                "sdnn_index_ms", "total_power_ms2", "lf_nu")
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT baseline FROM users WHERE user_id=?",
                               (self.user_id,)).fetchone()
            base = json.loads(row["baseline"]) if row else {}
            for k in keys:
                v = feats.get(k)
                if v is None or float(v) <= 0:
                    continue
                v = float(v)
                base[k] = round(v, 3) if k not in base or not base[k] \
                    else round(base[k] * (1 - alpha) + v * alpha, 3)
            base["updated_at"] = time.time()
            base["n_updates"] = int(base.get("n_updates", 0)) + 1
            conn.execute("UPDATE users SET baseline=? WHERE user_id=?",
                         (json.dumps(base), self.user_id))
            conn.commit()
        return True

    # ==================================================================
    # 互动记忆（v1 接口保留）
    # ==================================================================
    def save_interaction(self, message: str, response: str,
                         emotion_state: Optional[dict] = None, ts: float = None,
                         fta_id: Optional[int] = None):
        """保存一次互动（episodic 记忆）"""
        ts = ts if ts is not None else time.time()
        state = emotion_state or {}
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO interactions(user_id,ts,role,content,emotion,risk,fta_id)"
                " VALUES(?,?,?,?,?,?,?)",
                (self.user_id, ts, "user", message,
                 state.get("emotion"), state.get("risk_score"), fta_id))
            conn.execute(
                "INSERT INTO interactions(user_id,ts,role,content,emotion,risk,fta_id)"
                " VALUES(?,?,?,?,?,?,?)",
                (self.user_id, ts + 0.001, "assistant", response,
                 state.get("emotion"), state.get("risk_score"), fta_id))
            conn.commit()
            return cur.lastrowid

    def get_recent_interactions(self, limit: int = None) -> List[dict]:
        limit = limit or config.RECENT_CONTEXT_N
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT ts, role, content, emotion FROM interactions"
                " WHERE user_id=? ORDER BY ts DESC LIMIT ?",
                (self.user_id, limit * 2)).fetchall()
        out = [{"ts": datetime.fromtimestamp(r["ts"]).strftime("%m-%d %H:%M"),
                "role": r["role"], "content": r["content"],
                "emotion": r["emotion"]} for r in reversed(rows)]
        return out[-limit:]

    # ==================================================================
    # 语义检索（v2: 混合打分）
    # ==================================================================
    def semantic_search(self, query: str, top_k: int = None) -> List[dict]:
        """语义搜索历史互动（ChromaDB 向量检索 → 关键词打分兜底）"""
        top_k = top_k or config.MEMORY_TOP_K
        if self._chroma:
            try:
                res = self._chroma.query(query_texts=[query], n_results=top_k)
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                return [{"content": d, "meta": m or {}}
                        for d, m in zip(docs, metas) if d]
            except Exception:  # noqa: BLE001
                pass
        tokens = _tokenize(query)
        if not tokens:
            return []
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT ts, role, content, emotion FROM interactions"
                " WHERE user_id=? ORDER BY ts DESC LIMIT 200",
                (self.user_id,)).fetchall()
        scored = []
        for r in rows:
            score = sum(1 for t in tokens if t and t in (r["content"] or ""))
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        return [{"ts": s[1]["ts"], "role": s[1]["role"],
                 "content": s[1]["content"], "emotion": s[1]["emotion"],
                 "score": s[0]} for s in scored[:top_k]]

    # ==================================================================
    # FTA 情境记忆单元（v2 核心）
    # ==================================================================
    def save_fta_unit(self, fact_text: str,
                      situation_type: str = "general_fact",
                      emotion: str = "", emotion_conf: float = 0.0,
                      time_orientation: str = "unknown",
                      event_time: str = "", participants: str = "",
                      intention: str = "", relation_hint: str = "",
                      physio: Optional[dict] = None,
                      evidence_turn_id: Optional[int] = None,
                      evidence_span: str = "",
                      completeness: str = "complete",
                      lifecycle: str = "pending",
                      created_ts: Optional[float] = None,
                      link_to: Optional[Sequence[int]] = None,
                      link_relation: str = "followup_of") -> int:
        """写入一条情境级记忆单元（文献5 五元组）

        Args:
            fact_text: 事实锚 —— 这段对话讲了什么（一句话，情境级）
            situation_type: 见 ``CONTENT_TYPES``
            emotion: 情感锚的情绪标签
            emotion_conf: **情绪置信度**。文献5 的人工审计发现情感锚是
                噪声最大的部分（0.32 分/2.0，最低），故必须带置信度，
                低置信时下游应当**试探性确认**而非武断断言。
            time_orientation: past/present/future/mixed/unknown
            event_time: 事件发生时间（自然语言，如"上周三"）
            physio: **本项目原创的生理锚** —— 该记忆形成时的 HRV 快照
            completeness: partial 单元的会作为下一轮的"构造上下文"携带
            link_to: 需要建立时序链接的历史单元 id 列表

        Returns:
            新单元 id（partial 融合时返回被融合单元的 id）
        """
        created_ts = created_ts or time.time()
        if situation_type not in CONTENT_TYPES:
            situation_type = "general_fact"
        if time_orientation not in TIME_ORIENTATIONS:
            time_orientation = "unknown"
        if completeness not in COMPLETENESS:
            completeness = "complete"
        if lifecycle not in LIFECYCLE:
            lifecycle = "pending"

        with self._lock, self._conn() as conn:
            # ---- 式(4)(5)：partial 单元先尝试融合，而非新开一条 ----
            if config.MEMORY_FTA_ENABLED and completeness == "complete":
                pend = conn.execute(
                    "SELECT id, fact_text, situation_type FROM fta_units"
                    " WHERE user_id=? AND completeness='partial'"
                    " ORDER BY created_ts DESC LIMIT 3",
                    (self.user_id,)).fetchall()
                for p in pend:
                    if _similar(p["fact_text"], fact_text) >= config.MEMORY_FUSE_OVERLAP:
                        merged = f"{p['fact_text']}；{fact_text}".strip("；")
                        # ⚠️ emotion_conf 必须与 emotion **同步**更新。
                        # 只补 emotion 而不补置信度，会让一条"刚被用户亲口
                        # 确认"的情感锚在库里带着 0.0 的置信度 —— 下游
                        # recall_tool 会把低置信的情感标注为"仅供参考"，
                        # 等于把最可靠的一次证据显示成最不可靠的一次。
                        # NULLIF(?,'') 为空时保持原值，避免用空标签覆盖。
                        conn.execute(
                            "UPDATE fta_units SET fact_text=?,"
                            " completeness='complete',"
                            " emotion=COALESCE(NULLIF(?,''), emotion),"
                            " emotion_conf=CASE WHEN NULLIF(?,'') IS NULL"
                            "   THEN emotion_conf ELSE ? END,"
                            " physio_json=COALESCE(NULLIF(?,''), physio_json)"
                            " WHERE id=?",
                            (merged, emotion, emotion, emotion_conf,
                             json.dumps(physio, ensure_ascii=False)
                             if physio else "",
                             p["id"]))
                        conn.commit()
                        logger.info("FTA partial 单元融合 → id=%s", p["id"])
                        return int(p["id"])

            cur = conn.execute(
                "INSERT INTO fta_units(user_id, created_ts, fact_text,"
                " situation_type, participants, dialogue_ts, event_time,"
                " time_orientation, status, emotion, emotion_conf, intention,"
                " relation_hint, physio_json, evidence_turn_id, evidence_span,"
                " completeness, lifecycle, strength, last_recall_ts, recall_count)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                (self.user_id, created_ts, fact_text, situation_type,
                 participants, created_ts, event_time, time_orientation,
                 "active", emotion, emotion_conf, intention, relation_hint,
                 json.dumps(physio, ensure_ascii=False) if physio else None,
                 evidence_turn_id, evidence_span, completeness, lifecycle,
                 config.MEMORY_STRENGTH_INIT, created_ts))
            new_id = int(cur.lastrowid)
            conn.commit()

        # ---- 式(6)：建立时序链接 ----
        if link_to:
            for dst in link_to:
                self.link_units(new_id, int(dst), link_relation)
        return new_id

    def link_units(self, src_id: int, dst_id: int, relation: str,
                   confidence: float = 0.6) -> bool:
        """建立记忆间时序关系（文献5 式(6)）

        关系类型：updates / contradicts / supports / followup_of /
                 same_episode_as
        这是"上次你说要联系导师，后来怎么样了？"能问得出来的基础。
        """
        if relation not in LINK_RELATIONS:
            relation = "unknown"
        if src_id == dst_id:
            return False
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO memory_links(user_id, src_id, dst_id, relation,"
                " confidence, created_ts) VALUES(?,?,?,?,?,?)",
                (self.user_id, src_id, dst_id, relation, confidence, time.time()))
            conn.commit()
        return True

    def get_linked(self, unit_id: int, limit: int = 5) -> List[dict]:
        """取某单元的关联记忆（双向），供"追问后续"使用"""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT l.relation, l.confidence,"
                " f.id, f.fact_text, f.emotion, f.created_ts, f.lifecycle"
                " FROM memory_links l JOIN fta_units f"
                " ON (f.id = l.dst_id OR f.id = l.src_id)"
                " WHERE l.user_id=? AND (l.src_id=? OR l.dst_id=?)"
                "   AND f.id <> ?"
                " ORDER BY l.created_ts DESC LIMIT ?",
                (self.user_id, unit_id, unit_id, unit_id, limit)).fetchall()
        return [{"id": r["id"], "fact": r["fact_text"], "emotion": r["emotion"],
                 "relation": r["relation"], "confidence": r["confidence"],
                 "lifecycle": r["lifecycle"],
                 "ts": _fmt_ts(r["created_ts"])} for r in rows]

    def get_pending_units(self, limit: int = 3) -> List[dict]:
        """待补全的 partial 单元（式(4) 跨段携带）"""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id, fact_text, situation_type, created_ts FROM fta_units"
                " WHERE user_id=? AND completeness='partial' AND archived=0"
                " ORDER BY created_ts DESC LIMIT ?",
                (self.user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------- 遗忘曲线
    @staticmethod
    def retention(strength: float, elapsed_sec: float) -> float:
        """艾宾浩斯保留度 R = e^(−t/S)（文献4 MemoryBank）

        ⚠️ 文献4 自述这是"高度简化、探索性"的工程类比，**不是心理学定论**，
        引用时须注明。本项目的工程含义：把"多久没人提这件事"折算成
        0~1 的权重，用于排序与归档，而不是真的"遗忘"（数据仍在库里）。

        Args:
            strength: 记忆强度 S（每被召回一次 +1）
            elapsed_sec: 距上次召回/创建的时间（秒）
        """
        if strength <= 0:
            return 0.0
        # 以天为单位做归一，避免秒级数值下 e^(-t/S) 恒为 0
        t_days = max(0.0, elapsed_sec) / 86400.0
        return math.exp(-t_days / max(0.5, strength))

    def touch_recall(self, unit_ids: Sequence[int]):
        """标记记忆被召回：S += 1 且重置计时 —— "复述使遗忘曲线变平"

        这是文献4 的关键机制：常用记忆越来越"牢固"，冷门记忆自然下沉，
        不需要手工维护优先级。
        """
        if not config.MEMORY_FORGET_ENABLED or not unit_ids:
            return
        now = time.time()
        with self._lock, self._conn() as conn:
            for uid in unit_ids:
                conn.execute(
                    "UPDATE fta_units SET strength = strength + ?,"
                    " last_recall_ts = ?, recall_count = recall_count + 1"
                    " WHERE id=? AND user_id=?",
                    (config.MEMORY_STRENGTH_GAIN, now, int(uid), self.user_id))
            conn.commit()

    def archive_stale_units(self, floor: Optional[float] = None) -> int:
        """归档保留度过低的单元（**不删除**，便于调试与合规审计）

        依据实测经验，心理陪伴场景中"用户很久没再提起"本身就是信息，
        直接删除会丢失"这件事后来没再出现"这一事实。
        """
        floor = config.MEMORY_RETENTION_FLOOR if floor is None else floor
        now = time.time()
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id, strength, COALESCE(last_recall_ts, created_ts) AS t"
                " FROM fta_units WHERE user_id=? AND archived=0",
                (self.user_id,)).fetchall()
            stale = [r["id"] for r in rows
                     if self.retention(r["strength"], now - r["t"]) < floor]
            for uid in stale:
                conn.execute("UPDATE fta_units SET archived=1,"
                             " lifecycle='archived' WHERE id=?", (uid,))
            conn.commit()
        return len(stale)

    # ------------------------------------------------------- 混合检索
    def retrieve_fta(self, query: str, top_k: int = None,
                     tau: float = None, expand_links: bool = True,
                     only_active: bool = True) -> List[dict]:
        """FTA 记忆检索（文献5 式(8)(9)(10)）

        打分公式（原文式8）::

            R(q, m) = λ · s_emb(q, ρ(m)) + (1 − λ) · c(q, m)

        * ``s_emb``：语义相似度。本项目在**无嵌入模型**时用中文 bigram
          Jaccard 相似度作为代理（有 ChromaDB/text2vec 时自动切到真向量），
          并在返回结果的 ``emb_backend`` 中标明实际使用的后端 —— 避免
          "看起来是向量检索其实不是"的含混表述。
        * ``c``：**结构化线索匹配分**，对时间、情境类型、参与者、情感
          上下文做匹配。文献5 的消融实验表明**去掉时间信息退化最大**，
          所以这里时间线索权重最高。
        * λ = 0.8、τ = 0.35、K 见 config（与文献5 取值一致）。
        * ``expand_links``：式(10) 的链接邻居扩展，每个命中单元再带
          1 个关联记忆 —— 这是"追问后续"能力的来源。
        """
        top_k = top_k or config.MEMORY_FTA_TOP_K
        tau = config.MEMORY_RETRIEVAL_TAU if tau is None else tau
        lam = config.MEMORY_RETRIEVAL_LAMBDA
        if not config.MEMORY_FTA_ENABLED:
            return []

        emb_backend = "bigram" if self._chroma is None else "chroma"
        sql = ("SELECT * FROM fta_units WHERE user_id=?"
               + (" AND archived=0" if only_active else "")
               + " ORDER BY created_ts DESC LIMIT 300")
        with self._lock, self._conn() as conn:
            rows = conn.execute(sql, (self.user_id,)).fetchall()
        if not rows:
            return []

        q_tokens = _tokenize(query)
        now = time.time()
        scored: List[Tuple[float, dict]] = []
        for r in rows:
            unit = dict(r)
            doc = " ".join(str(unit.get(k) or "") for k in
                           ("fact_text", "situation_type", "participants",
                            "emotion", "intention", "relation_hint",
                            "event_time"))
            s_emb = _similar(query, doc, q_tokens)
            c_struct = self._structured_score(unit, query, now)
            score = lam * s_emb + (1 - lam) * c_struct
            # 保留度作为**弱权重**参与排序：文献4 的遗忘曲线在此生效
            if config.MEMORY_FORGET_ENABLED:
                r_keep = self.retention(unit.get("strength") or 1.0,
                                        now - (unit.get("last_recall_ts")
                                               or unit.get("created_ts") or now))
                score *= (0.75 + 0.25 * r_keep)
            if score >= tau:
                scored.append((score, unit))

        scored.sort(key=lambda x: -x[0])
        picked = scored[:top_k]
        if not picked:
            return []

        self.touch_recall([u["id"] for _, u in picked])

        out: List[dict] = []
        for score, u in picked:
            item = {"id": u["id"], "fact": u["fact_text"],
                    "type": u["situation_type"], "emotion": u["emotion"],
                    "emotion_conf": u["emotion_conf"],
                    "time_orientation": u["time_orientation"],
                    "event_time": u["event_time"],
                    "lifecycle": u["lifecycle"],
                    "score": round(score, 4),
                    "ts": _fmt_ts(u["created_ts"]),
                    "emb_backend": emb_backend,
                    "physio": (json.loads(u["physio_json"])
                               if u.get("physio_json") else None)}
            if expand_links:
                item["linked"] = self.get_linked(u["id"], limit=1)
            out.append(item)
        return out

    @staticmethod
    def _structured_score(unit: dict, query: str, now: float) -> float:
        """结构化线索匹配分 c(q,m)（文献5 式(8) 的第二项）

        权重设计依据文献5 的消融结果：**时间信息退化最大**（权重最高），
        其次是情感信息，关系链接影响最小。
        """
        score = 0.0
        # 1) 时间线索（0.40）—— 越近的记忆越可能相关
        ts = unit.get("created_ts") or now
        age_days = max(0.0, (now - ts) / 86400.0)
        score += 0.40 * math.exp(-age_days / 14.0)     # 14 天半衰
        # 2) 情感线索（0.25）—— 查询里出现情绪词或单元带情绪
        emo = unit.get("emotion") or ""
        if emo and (emo in query or any(k in query for k in ("难受", "焦虑",
                                                             "压力", "开心",
                                                             "累", "紧张"))):
            score += 0.25
        else:
            score += 0.08
        # 3) 情境类型线索（0.20）—— 风险与待办类记忆优先被想起
        stype = unit.get("situation_type") or ""
        if stype == "risk_signal":
            score += 0.20
        elif stype == "homework_or_plan":
            score += 0.16
        elif stype in ("current_state", "support_interaction"):
            score += 0.12
        else:
            score += 0.06
        # 4) 完成度与生命周期（0.15）—— 未完成/被推翻的事更需要追问
        if unit.get("completeness") == "partial":
            score += 0.10
        if unit.get("lifecycle") in ("pending", "contradicted"):
            score += 0.05
        return min(1.0, score)

    # ==================================================================
    # 分层摘要与人格画像（文献4 MemoryBank）
    # ==================================================================
    def upsert_daily_summary(self, day: str, summary: str,
                             emotion_stat: Optional[dict] = None):
        """写入日摘要（分层摘要的中层）"""
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO daily_summary(user_id, day, summary, emotion_stat,"
                " updated_ts) VALUES(?,?,?,?,?)"
                " ON CONFLICT(user_id, day) DO UPDATE SET"
                " summary=excluded.summary, emotion_stat=excluded.emotion_stat,"
                " updated_ts=excluded.updated_ts",
                (self.user_id, day, summary,
                 json.dumps(emotion_stat or {}, ensure_ascii=False), time.time()))
            conn.commit()

    def get_daily_summaries(self, days: int = 7) -> List[dict]:
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT day, summary, emotion_stat FROM daily_summary"
                " WHERE user_id=? AND day>=? ORDER BY day DESC",
                (self.user_id, since)).fetchall()
        return [{"day": r["day"], "summary": r["summary"],
                 "emotion_stat": json.loads(r["emotion_stat"] or "{}")}
                for r in rows]

    def update_portrait(self, portrait: str, merge: bool = False):
        """更新动态人格画像（分层摘要的高层，文献4 的 personality understanding）

        与 ``users.profile`` 的区别：profile 存**用户显式提供**的事实
        （称呼、偏好），portrait 存**系统推断**的人格与情绪倾向，
        两者分开存放，避免"推断把事实覆盖掉"。
        """
        with self._lock, self._conn() as conn:
            if not merge:
                conn.execute(
                    "INSERT INTO user_portrait(user_id, portrait, updated_ts)"
                    " VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET"
                    " portrait=excluded.portrait, updated_ts=excluded.updated_ts",
                    (self.user_id, portrait, time.time()))
            else:
                row = conn.execute("SELECT portrait FROM user_portrait"
                                   " WHERE user_id=?", (self.user_id,)).fetchone()
                old = (row["portrait"] if row else "") or ""
                merged = (old + "\n" + portrait).strip() if old else portrait
                conn.execute(
                    "INSERT INTO user_portrait(user_id, portrait, updated_ts)"
                    " VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET"
                    " portrait=excluded.portrait, updated_ts=excluded.updated_ts",
                    (self.user_id, merged, time.time()))
            conn.commit()

    def get_portrait(self) -> str:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT portrait FROM user_portrait"
                               " WHERE user_id=?", (self.user_id,)).fetchone()
        return (row["portrait"] if row and row["portrait"] else "") or ""

    # ==================================================================
    # 情绪时间线（v1 接口保留 + v2 扩展字段）
    # ==================================================================
    def record_emotion_state(self, state: dict, ts: float = None):
        """记录一次情绪状态（pattern 记忆）

        v2 增记 jitter / 质量等级 / PAD 三维，使"当时的信号可信度"
        可回溯 —— 否则事后无法分辨某段低落在生理上是否可信。
        """
        ts = ts if ts is not None else time.time()
        pad = state.get("pad") or {}
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO emotion_timeline"
                "(user_id,ts,emotion,confidence,risk,hr_bpm,sdnn_ms,rmssd_ms,"
                " lf_hf,layer,jitter,quality_level,pnn50,sd_hr_bpm,"
                " pad_p,pad_a,pad_d)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.user_id, ts, state.get("emotion", ""),
                 state.get("confidence"), state.get("risk_score"),
                 state.get("hr_bpm"), state.get("sdnn_ms"),
                 state.get("rmssd_ms"), state.get("lf_hf"),
                 state.get("layer"), state.get("jitter"),
                 state.get("quality_level"), state.get("pnn50"),
                 state.get("sd_hr_bpm"),
                 pad.get("P"), pad.get("A"), pad.get("D")))
            conn.commit()

    def get_emotional_timeline(self, days: float = 7) -> List[dict]:
        """获取最近 N 天情绪时间线"""
        since = time.time() - days * 86400
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT ts, emotion, confidence, risk, hr_bpm, sdnn_ms,"
                " rmssd_ms, lf_hf, layer, jitter, quality_level,"
                " pad_p, pad_a, pad_d FROM emotion_timeline"
                " WHERE user_id=? AND ts>=? ORDER BY ts ASC",
                (self.user_id, since)).fetchall()
        return [{"ts": datetime.fromtimestamp(r["ts"]).strftime("%m-%d %H:%M"),
                 "ts_epoch": r["ts"], "emotion": r["emotion"],
                 "confidence": r["confidence"], "risk": r["risk"],
                 "hr_bpm": r["hr_bpm"], "sdnn_ms": r["sdnn_ms"],
                 "rmssd_ms": r["rmssd_ms"], "lf_hf": r["lf_hf"],
                 "layer": r["layer"], "jitter": r["jitter"],
                 "quality_level": r["quality_level"],
                 "pad": {"P": r["pad_p"], "A": r["pad_a"], "D": r["pad_d"]}}
                for r in rows]

    def get_risk_history(self, limit: int = 20) -> List[float]:
        """最近 N 个风险评分（触发引擎用）"""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT risk FROM emotion_timeline WHERE user_id=?"
                " ORDER BY ts DESC LIMIT ?", (self.user_id, limit)).fetchall()
        return [float(r["risk"]) for r in reversed(rows) if r["risk"] is not None]

    def emotion_distribution(self, days: float = 7) -> Dict[str, int]:
        """情绪分布统计（供日摘要与画像更新使用）"""
        since = time.time() - days * 86400
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT emotion, COUNT(*) AS n FROM emotion_timeline"
                " WHERE user_id=? AND ts>=? GROUP BY emotion",
                (self.user_id, since)).fetchall()
        return {r["emotion"] or "未知": r["n"] for r in rows}

    # ==================================================================
    # 反馈记录（v1 接口保留）
    # ==================================================================
    def record_feedback(self, strategy: str, before_risk: float,
                        after_risk: float, effective: bool):
        """记录干预前后风险对比（STEP 06 反馈优化）"""
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO feedback(user_id,ts,strategy,before_risk,"
                "after_risk,effective) VALUES(?,?,?,?,?,?)",
                (self.user_id, time.time(), strategy, before_risk,
                 after_risk, 1 if effective else 0))
            conn.commit()

    def strategy_stats_persisted(self) -> Dict[str, Dict[str, int]]:
        """从库中重建策略有效率统计（进程重启后不丢失）"""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT strategy, COUNT(*) AS total, SUM(effective) AS eff"
                " FROM feedback WHERE user_id=? GROUP BY strategy",
                (self.user_id,)).fetchall()
        return {r["strategy"]: {"total": int(r["total"]),
                                "effective": int(r["eff"] or 0)}
                for r in rows if r["strategy"]}

    # ==================================================================
    # 主观自评（v2：大创实验真值来源 + 主客观校准）
    # ==================================================================
    def save_self_report(self, score: float, emotion: str = "", note: str = "",
                         objective: Optional[dict] = None,
                         ts: Optional[float] = None) -> int:
        """保存一次主观自评，并把它与**同一时刻的客观侧快照**配对存下

        为什么必须成对存：情绪识别准确率这个指标只有在
        「(客观判定, 主观真值)」成对存在时才算得出来。分开存会让
        事后对齐变成一件很容易出错的事（时间戳漂移、多设备时钟不同步）。

        Args:
            score: 主观不适程度 0~10（越大越难受）
            emotion: 主观情绪标签（可选，用户自己的说法优先于系统术语）
            note: 主观补充描述
            objective: 客观侧快照 dict（emotion/confidence/risk/pad/quality…）
            ts: 时间戳（默认当前）

        Returns:
            新记录 id
        """
        ts = ts if ts is not None else time.time()
        obj = objective or {}
        if 'simulation' in obj and (obj.get('simulation') is not False or not obj.get('validated')):
            obj = {**obj, 'emotion': '', 'confidence': None, 'risk_score': None, 'pad': None,
                   'quality': {**(obj.get('quality') or {}), 'quality_level': '未分类'}}
        emo_obj = obj.get("emotion") or ""
        # 主观分数 → 客观风险的可比尺度（0~10 → 0~100）
        gap = 0.0
        conflict = 0
        try:
            if score is not None and obj.get("risk_score") is not None and obj.get('simulation') is not False:
                gap = abs(float(score) * 10.0 - float(obj.get("risk_score") or 0))
                conflict = 1 if gap >= config.SELF_REPORT_CONFLICT_GAP * 10 else 0
        except (TypeError, ValueError):
            gap, conflict = 0.0, 0
        # 主观标签与客观判定不一致，也算一种冲突信号
        if emotion and emo_obj and emotion != emo_obj:
            conflict = 1
            gap = max(gap, config.SELF_REPORT_CONFLICT_GAP * 10)

        pad = obj.get("pad") or {}
        quality = obj.get("quality") or {}
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO self_report(user_id, ts, score, emotion, note,"
                " o_emotion, o_confidence, o_risk, o_pad_p, o_pad_a, o_pad_d,"
                " o_jitter, o_quality, conflict, gap)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.user_id, ts, score, emotion, note, emo_obj,
                 obj.get("confidence"), obj.get("risk_score"),
                 pad.get("P"), pad.get("A"), pad.get("D"),
                 quality.get("jitter"), quality.get("quality_level"),
                 conflict, round(gap, 2)))
            conn.commit()
            return int(cur.lastrowid or 0)

    def get_self_reports(self, limit: int = 20) -> List[dict]:
        """最近 N 条主观自评（含客观对照）"""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM self_report WHERE user_id=?"
                " ORDER BY ts DESC LIMIT ?", (self.user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def self_report_consistency(self) -> dict:
        """主客观一致性统计（既用于对话降权，也用于实验报告）

        Returns:
            {"n": 样本数, "conflict_rate": 冲突率, "mean_gap": 平均差距,
             "n_conflict": 冲突条数, "n_reliable": 客观侧信号合格条数}
        """
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n,"
                " SUM(CASE WHEN o_emotion<>'' THEN 1 ELSE 0 END) AS ncomp,"
                " SUM(CASE WHEN o_emotion<>'' THEN conflict ELSE 0 END) AS nc,"
                " AVG(CASE WHEN o_emotion<>'' THEN gap ELSE NULL END) AS mg,"
                " SUM(CASE WHEN o_emotion<>'' AND o_quality IN ('合格','good') THEN 1 ELSE 0 END)"
                " AS nrel"
                " FROM self_report WHERE user_id=?",
                (self.user_id,)).fetchone()
        n = int(row["n"] or 0) if row else 0
        nc = int(row["nc"] or 0) if row else 0
        ncomp = int(row['ncomp'] or 0) if row else 0
        return {"n": n,
                "n_comparable": ncomp,
                "n_conflict": nc,
                "conflict_rate": round(nc / ncomp, 4) if ncomp else 0.0,
                "mean_gap": round(float(row["mg"] or 0.0), 2) if row else 0.0,
                "n_reliable": int(row["nrel"] or 0) if row else 0}

    # ==================================================================
    # 数据权利（文献5 伦理要求：查看 / 更正 / 删除 / 留存控制）
    # ==================================================================
    def export_user_data(self) -> dict:
        """导出该用户的全部数据（数据可携带权）

        文献5 的伦理声明明确要求：长期记忆会引入跨会话持续影响的风险，
        部署必须提供**查看/更正/删除/留存控制**。本方法支撑"查看"。
        """
        with self._lock, self._conn() as conn:
            def _all(sql, args=()):
                return [dict(r) for r in conn.execute(sql, args).fetchall()]

            # 本方法已在持锁状态下。锁为 RLock（可重入），因此调用
            # get_user_profile() 之类的方法**不会**死锁 —— 但为了减少
            # 重复查询，绝大多数数据仍直接用当前连接取。
            row_u = conn.execute("SELECT profile, baseline FROM users"
                                 " WHERE user_id=?", (self.user_id,)).fetchone()
            row_p = conn.execute("SELECT portrait FROM user_portrait"
                                 " WHERE user_id=?", (self.user_id,)).fetchone()
            return {
                "user_id": self.user_id,
                "exported_at": time.time(),
                "profile": json.loads(row_u["profile"]) if row_u else {},
                "baseline": json.loads(row_u["baseline"]) if row_u else {},
                "portrait": (row_p["portrait"] if row_p else "") or "",
                "interactions": _all(
                    "SELECT ts,role,content,emotion,risk,fta_id FROM interactions"
                    " WHERE user_id=? ORDER BY ts", (self.user_id,)),
                "emotion_timeline": _all(
                    "SELECT * FROM emotion_timeline WHERE user_id=?"
                    " ORDER BY ts", (self.user_id,)),
                "fta_units": _all(
                    "SELECT * FROM fta_units WHERE user_id=?"
                    " ORDER BY created_ts", (self.user_id,)),
                "memory_links": _all(
                    "SELECT * FROM memory_links WHERE user_id=?",
                    (self.user_id,)),
                "daily_summary": _all(
                    "SELECT * FROM daily_summary WHERE user_id=?",
                    (self.user_id,)),
                "feedback": _all(
                    "SELECT * FROM feedback WHERE user_id=?", (self.user_id,)),
                "self_report": _all(
                    "SELECT * FROM self_report WHERE user_id=?"
                    " ORDER BY ts", (self.user_id,)),
                "self_report_consistency": self.self_report_consistency(),
            }

    def purge_user_data(self, keep_profile: bool = True) -> dict:
        """删除该用户的记忆数据（被遗忘权）

        Args:
            keep_profile: True 时保留 profile/baseline（避免"删干净后
                基线归零导致首次识别失真"），仅清空对话与情境记忆。

        Returns:
            各表删除行数统计
        """
        tables = ["interactions", "emotion_timeline", "fta_units",
                  "memory_links", "daily_summary", "feedback", "self_report"]
        counts: Dict[str, int] = {}
        with self._lock, self._conn() as conn:
            for t in tables:
                cur = conn.execute(f"DELETE FROM {t} WHERE user_id=?",
                                   (self.user_id,))
                counts[t] = cur.rowcount
            conn.execute("DELETE FROM user_portrait WHERE user_id=?",
                         (self.user_id,))
            counts["user_portrait"] = 1
            if not keep_profile:
                conn.execute("DELETE FROM users WHERE user_id=?",
                             (self.user_id,))
                conn.execute("INSERT OR IGNORE INTO users(user_id, created_at)"
                             " VALUES(?,?)", (self.user_id, time.time()))
                counts["users"] = 1
            conn.commit()
        if self._chroma:
            try:
                self._chroma.delete(where={"kind": "interaction"})
            except Exception:  # noqa: BLE001
                pass
        return counts

    def update_fta_unit(self, unit_id: int, **fields) -> bool:
        """更正某条记忆（被纠正权 —— 用户说"你记错了"时的落地路径）"""
        allowed = {"fact_text", "situation_type", "emotion", "emotion_conf",
                   "time_orientation", "event_time", "participants",
                   "intention", "relation_hint", "completeness", "lifecycle",
                   "status"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                vals.append(v)
        if not sets:
            return False
        vals.extend([int(unit_id), self.user_id])
        with self._lock, self._conn() as conn:
            conn.execute(f"UPDATE fta_units SET {', '.join(sets)}"
                         f" WHERE id=? AND user_id=?", vals)
            conn.commit()
        return True

    # ==================================================================
    # 统计
    # ==================================================================
    def stats(self) -> dict:
        with self._lock, self._conn() as conn:
            n_int = conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE user_id=?",
                (self.user_id,)).fetchone()[0]
            n_emo = conn.execute(
                "SELECT COUNT(*) FROM emotion_timeline WHERE user_id=?",
                (self.user_id,)).fetchone()[0]
            n_fta = conn.execute(
                "SELECT COUNT(*) FROM fta_units WHERE user_id=?",
                (self.user_id,)).fetchone()[0]
            n_arch = conn.execute(
                "SELECT COUNT(*) FROM fta_units WHERE user_id=? AND archived=1",
                (self.user_id,)).fetchone()[0]
            n_link = conn.execute(
                "SELECT COUNT(*) FROM memory_links WHERE user_id=?",
                (self.user_id,)).fetchone()[0]
        return {"interactions": n_int, "emotion_records": n_emo,
                "fta_units": n_fta, "fta_archived": n_arch,
                "memory_links": n_link,
                "profile_keys": len(self.get_user_profile()),
                "baseline": self.get_baseline()}

    def close(self):
        try:
            self._db_conn.close()
        except Exception:  # noqa: BLE001
            pass


# ======================================================================
# 文本相似度与切分辅助（无嵌入模型时的轻量代理）
# ======================================================================
def _tokenize(text: str) -> set:
    """中文分词代理：相邻 bigram + 英文/数字词。

    ⚠️ **为什么只用 bigram 而不用单字**：
    本函数产出的集合要参与"重叠系数"计算（见 ``_similar``）。若混入单字，
    query「考试」与任意含"考"的文档会得到 1/2 的高分，召回噪声极大。
    只用 bigram（如「考试」「压力」）能显著降低假阳性。

    有 ChromaDB（真向量后端）时，本函数仅用于结构化兜底打分，
    届时 ``s_emb`` 会改用文本嵌入，不再依赖此代理。
    """
    if not text:
        return set()
    # 去掉标点与空白，只保留汉字/字母/数字 —— 否则 bigram 会被标点污染
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(text))
    if not t:
        return set()
    toks = {t[i:i + 2] for i in range(len(t) - 1)}
    toks |= set(re.findall(r"[A-Za-z0-9]{2,}", t))
    return toks


def _similar(a: str, b: str, a_tokens: Optional[set] = None) -> float:
    """重叠系数（Szymkiewicz–Simpson）作为语义相似度代理。

    定义: ``overlap = |A ∩ B| / min(|A|, |B|)``

    **为什么不用 Jaccard**：检索场景是"短 query → 长文档"，Jaccard 的分母
    是并集，会被长文档稀释到 0.05~0.15 量级；而文献5 式(8) 的 λ=0.8 是针对
    **余弦相似度**（典型 0.5~0.9）标定的，直接套用会导致量纲失配、
    检索几乎总是返回空。重叠系数衡量"短集合被长集合覆盖的比例"，
    量纲与余弦接近，才配得上文献给的 λ 与 τ。

    （实测：同一 query，Jaccard ≈ 0.08 → 综合分 0.22 < τ=0.35 召回为空；
      重叠系数 ≈ 0.67 → 综合分 0.69，正常召回。）

    ⚠️ **能力边界（必须如实告知，勿过度宣称）**：本代理只看字符 bigram，
    因此"语义相关但用词完全不同"的两句会得 0 分 —— 例如
    「在准备一个考试」与「下周的高数补考」重叠为 0，
    既不会触发 partial 融合，也不会在检索里互相召回。
    这是**无嵌入模型时的固有限制**，不是参数没调好。
    升级路径：``CHROMA_ENABLED=1`` 接真向量后端，或用 LLM 做边界检测；
    在此之前，本模块的语义能力只应描述为"词面重合检索"。
    """
    ta = a_tokens if a_tokens is not None else _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    return inter / min(len(ta), len(tb))


def _fmt_ts(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return ""


# ------------------------------------------------------------------ 自测
if __name__ == "__main__":
    print("=" * 66)
    print("长期记忆管理器 v2 自测（FTA-Mem + MemoryBank）")
    print("=" * 66)

    mm = MemoryManager("test_user", db_path=":memory:")
    mm.update_user_profile({"preferences": {"称呼": "同学", "话题": "考试压力"}})

    # --- 写入情境级记忆单元 ---
    id1 = mm.save_fta_unit(
        fact_text="用户提到下周有专业课期末考试，感到压力很大",
        situation_type="current_state", emotion="高焦虑", emotion_conf=0.8,
        time_orientation="future", event_time="下周",
        physio={"rmssd_ms": 18.0, "sdnn_ms": 24.0, "jitter": 0.05})
    id2 = mm.save_fta_unit(
        fact_text="用户决定每天晚饭后去操场散步半小时",
        situation_type="homework_or_plan", emotion="低焦虑", emotion_conf=0.6,
        time_orientation="future", link_to=[id1], link_relation="followup_of")
    id3 = mm.save_fta_unit(
        fact_text="用户说其实不是考试，是担心考不好让父母失望",
        situation_type="belief_candidate", emotion="低焦虑", emotion_conf=0.5,
        link_to=[id1], link_relation="updates")
    print(f"\n写入 3 条 FTA 单元: ids={id1},{id2},{id3}")

    # --- partial 融合 ---
    p1 = mm.save_fta_unit(fact_text="用户提到和室友的关系有点", completeness="partial")
    p2 = mm.save_fta_unit(fact_text="用户提到和室友的关系有点紧张，因为作息不同",
                          completeness="complete")
    print(f"partial 融合: 新建 id={p1} → 补全后复用 id={p2} "
          f"({'成功（未新增单元）' if p1 == p2 else '未融合'})")

    # --- 混合检索 ---
    print("\n混合检索 '考试压力':")
    for it in mm.retrieve_fta("考试压力"):
        print(f"  [{it['score']:.3f}] id={it['id']} {it['fact'][:30]}"
              f"  ({it['type']}, backend={it['emb_backend']})")
        for lk in it.get("linked", []):
            print(f"        └ {lk['relation']}: {lk['fact'][:26]}")

    # --- 遗忘曲线 ---
    print("\n遗忘曲线 R = e^(−t/S)（强度 S 随召回增长）:")
    for s in (1.0, 2.0, 5.0, 10.0):
        vals = [mm.retention(s, d * 86400) for d in (1, 7, 30, 90)]
        print(f"  S={s:<5} " + "  ".join(f"{d:>2}天:{v:.3f}"
                                        for d, v in zip((1, 7, 30, 90), vals)))

    print(f"\n召回后强度变化:")
    before = mm.retrieve_fta("考试")[0]["id"]
    print(f"  命中 id={before}（strength 已 +1）")

    # --- 数据权利 ---
    data = mm.export_user_data()
    print(f"\n数据导出: 互动 {len(data['interactions'])} 条, "
          f"FTA {len(data['fta_units'])} 条, "
          f"链接 {len(data['memory_links'])} 条")
    print(f"统计: {mm.stats()}")
