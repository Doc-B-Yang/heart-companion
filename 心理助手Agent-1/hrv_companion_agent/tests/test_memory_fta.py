# -*- coding: utf-8 -*-
"""FTA 情境记忆与主观自评单元测试（对应文献4 MemoryBank + 文献5 FTA-Mem）

文献5 的五个关键机制各有一组断言：
  ① **五元组**单元结构（事实锚 / 时间锚 / 情感锚 / 证据指针 / 构造状态）
  ② **partial 跨段携带与融合**（式4、5）—— 用户一句话说一半就走了，
     下次回来系统要能接上
  ③ **混合检索**（式8）：λ·语义 + (1−λ)·结构化，且要暴露实际用的后端
  ④ **时序链接**（式6）：updates / contradicts / followup_of
  ⑤ **遗忘曲线**（文献4）：R=e^(−t/S)，召回一次 S+1 并重置计时
另有一组覆盖 ``self_report`` 表：它承担的是"识别准确率"这个指标的
**真值来源**，所以"成对存"和"冲突判定"都必须钉死。
"""

import time
import unittest

from hrv_companion_agent import config
from hrv_companion_agent.agent.memory_manager import (COMPLETENESS, CONTENT_TYPES,
                                                      LIFECYCLE, LINK_RELATIONS,
                                                      TIME_ORIENTATIONS,
                                                      MemoryManager, _similar)


class FtaTestBase(unittest.TestCase):
    def setUp(self):
        self.mm = MemoryManager("fta_user", db_path=":memory:")

    def tearDown(self):
        try:
            self.mm.close()
        except Exception:  # noqa: BLE001
            pass


class TestUnitStructure(FtaTestBase):
    """五元组的每条锚都必须真的落库，而不是填了就忘"""

    def test_all_five_anchors_persist(self):
        uid = self.mm.save_fta_unit(
            fact_text="下周三要交大创中期报告，还没开始写",
            situation_type="homework_or_plan",
            emotion="低焦虑", emotion_conf=0.72,
            time_orientation="future", event_time="下周三",
            participants="导师、队友",
            intention="想提前两天写完",
            relation_hint="与导师的关系",
            physio={"rmssd_ms": 28.5, "jitter": 0.04},
            evidence_turn_id="turn_7", evidence_span="下周三要交中期报告",
            completeness="complete", lifecycle="pending")
        self.assertGreater(uid, 0)
        rows = self.mm.retrieve_fta("中期报告", top_k=5)
        self.assertTrue(rows, "刚写入的单元应当能被检索到")
        got = next(r for r in rows if r["id"] == uid)
        # 事实锚 / 时间锚 / 情感锚 / 证据指针 / 构造状态
        self.assertIn("中期报告", got["fact"])
        self.assertEqual(got["time_orientation"], "future")
        self.assertEqual(got["event_time"], "下周三")
        self.assertEqual(got["emotion"], "低焦虑")
        self.assertAlmostEqual(got["emotion_conf"], 0.72, places=6)
        self.assertEqual(got["lifecycle"], "pending")
        self.assertIsNotNone(got["physio"])
        self.assertAlmostEqual(got["physio"]["rmssd_ms"], 28.5, places=6)

    def test_evidence_pointer_is_stored(self):
        uid = self.mm.save_fta_unit("和室友吵了一架", situation_type="relationship_event",
                                    evidence_turn_id="turn_12",
                                    evidence_span="他半夜打游戏外放")
        data = self.mm.export_user_data()
        unit = next(u for u in data["fta_units"] if u["id"] == uid)
        self.assertEqual(unit["evidence_turn_id"], "turn_12")
        self.assertIn("外放", unit["evidence_span"])

    def test_invalid_enum_values_are_coerced(self):
        """非法枚举值必须被夹到合法集合内，不得写进库（否则下游解析会崩）"""
        uid = self.mm.save_fta_unit("随便一句", situation_type="不存在的类型",
                                    time_orientation="以前", completeness="半截",
                                    lifecycle="未知状态")
        unit = next(u for u in self.mm.export_user_data()["fta_units"]
                    if u["id"] == uid)
        self.assertIn(unit["situation_type"], CONTENT_TYPES)
        self.assertIn(unit["time_orientation"], TIME_ORIENTATIONS)
        self.assertIn(unit["completeness"], COMPLETENESS)
        self.assertIn(unit["lifecycle"], LIFECYCLE)

    def test_initial_strength_is_configured_value(self):
        uid = self.mm.save_fta_unit("初始强度测试", situation_type="event")
        unit = next(u for u in self.mm.export_user_data()["fta_units"]
                    if u["id"] == uid)
        self.assertAlmostEqual(unit["strength"],
                               config.MEMORY_STRENGTH_INIT, places=6)
        self.assertEqual(unit["recall_count"], 0)


class TestPartialCarryOver(FtaTestBase):
    """式(4)(5)：partial 跨段携带与融合 —— "上次没说完的事" """

    def test_partial_unit_is_listed_as_pending(self):
        uid = self.mm.save_fta_unit("我最近在准备一个考试",
                                    situation_type="event",
                                    completeness="partial")
        pending = self.mm.get_pending_units(limit=5)
        self.assertIn(uid, [p["id"] for p in pending])

    def test_complete_unit_fuses_into_partial(self):
        """后续补充信息应**融合进**原单元，而不是新开一条

        这是"记忆是连续的"与"记忆是一堆碎片"的分水岭：
        若新开一条，用户会看到系统把同一件事记住两遍。
        """
        first = self.mm.save_fta_unit("用户提到和室友的关系有点",
                                      situation_type="relationship_event",
                                      completeness="partial")
        second = self.mm.save_fta_unit("和室友关系有点紧张，因为作息不同",
                                       situation_type="relationship_event",
                                       completeness="complete")
        self.assertEqual(second, first, "complete 单元应融合进 partial 单元")
        units = self.mm.export_user_data()["fta_units"]
        self.assertEqual(len([u for u in units
                              if u["situation_type"] == "relationship_event"]), 1)
        merged = next(u for u in units if u["id"] == first)
        self.assertEqual(merged["completeness"], "complete")
        self.assertIn("作息不同", merged["fact_text"])

    def test_fusion_requires_lexical_continuity(self):
        """**能力边界**：语义相关但用词完全不同的两句**不会**融合

        「在准备一个考试」→「下周的高数补考」在人看来是同一件事的两次提及，
        但两者没有共同 bigram，``_similar`` 返回 0，故不融合。

        这条测试**故意把限制写进契约**，防止文档或汇报中出现
        "系统能自动把同一件事的两次提及连起来"这类不成立的表述。
        要覆盖这种情况需接真向量后端（CHROMA_ENABLED=1）或 LLM 边界检测。
        若哪天换了后端使这条测试失败，说明能力升级了 ——
        届时应当庆祝并更新文档，而不是改掉这个断言。
        """
        self.assertEqual(
            _similar("我最近在准备一个考试", "就是下周五的高数补考，还没复习"),
            0.0)
        first = self.mm.save_fta_unit("我最近在准备一个考试",
                                      situation_type="event",
                                      completeness="partial")
        second = self.mm.save_fta_unit("就是下周五的高数补考，还没复习",
                                       situation_type="event",
                                       completeness="complete")
        self.assertNotEqual(second, first)
        self.assertEqual(len(self.mm.export_user_data()["fta_units"]), 2)

    def test_fusion_threshold_is_configurable(self):
        """阈值取自配置而非魔法数字，且取值须与 τ 同量级"""
        self.assertAlmostEqual(config.MEMORY_FUSE_OVERLAP, 0.35, places=6)
        self.assertLessEqual(config.MEMORY_FUSE_OVERLAP, 1.0)
        self.assertGreater(config.MEMORY_FUSE_OVERLAP, 0.0)

    def test_unrelated_statement_does_not_fuse(self):
        """不相关内容不得被误融合（宁缺毋滥）"""
        first = self.mm.save_fta_unit("我最近在准备一个考试",
                                      situation_type="event",
                                      completeness="partial")
        second = self.mm.save_fta_unit("今天中午吃的是西红柿鸡蛋面",
                                       situation_type="general_fact",
                                       completeness="complete")
        self.assertNotEqual(second, first)
        self.assertEqual(len(self.mm.get_pending_units(limit=5)), 1)

    def test_emotion_filled_on_fusion(self):
        """融合时若原单元没有情感锚，应由补充信息补上"""
        first = self.mm.save_fta_unit("用户提到和室友的关系有点",
                                      situation_type="relationship_event",
                                      completeness="partial")
        self.mm.save_fta_unit("和室友关系有点紧张，挺焦虑的",
                              situation_type="relationship_event",
                              completeness="complete",
                              emotion="高焦虑", emotion_conf=0.8)
        unit = next(u for u in self.mm.export_user_data()["fta_units"]
                    if u["id"] == first)
        self.assertEqual(unit["emotion"], "高焦虑")
        self.assertAlmostEqual(unit["emotion_conf"], 0.8, places=6)


class TestTemporalLinks(FtaTestBase):
    """式(6)：记忆间时序关系 —— "后来怎么样了" 的技术基础"""

    def test_link_creation_on_insert(self):
        plan = self.mm.save_fta_unit("我打算这周给导师发邮件",
                                     situation_type="homework_or_plan")
        later = self.mm.save_fta_unit("邮件已经发出去了，还没回复",
                                      situation_type="event",
                                      link_to=[plan],
                                      link_relation="followup_of")
        linked = self.mm.get_linked(later, limit=5)
        self.assertTrue(linked)
        self.assertEqual(linked[0]["id"], plan)

    def test_link_is_bidirectional_lookup(self):
        """从任一端都应能查到另一端（查询实现是 src 或 dst 都命中）"""
        a = self.mm.save_fta_unit("想去校心理中心问问", situation_type="event")
        b = self.mm.save_fta_unit("预约已经取消了", situation_type="event",
                                  link_to=[a], link_relation="updates")
        self.assertEqual(self.mm.get_linked(a, limit=5)[0]["id"], b)
        self.assertEqual(self.mm.get_linked(b, limit=5)[0]["id"], a)

    def test_invalid_relation_falls_back(self):
        """非法关系名不得写进库（否则前端按关系名做分支时会静默失效）"""
        a = self.mm.save_fta_unit("甲", situation_type="event")
        b = self.mm.save_fta_unit("乙", situation_type="event",
                                  link_to=[a], link_relation="随便写的关系")
        rel = self.mm.get_linked(b, limit=5)[0]["relation"]
        self.assertIn(rel, LINK_RELATIONS)
        self.assertEqual(rel, "unknown")

    def test_self_link_rejected(self):
        a = self.mm.save_fta_unit("甲", situation_type="event")
        self.assertFalse(self.mm.link_units(a, a, "followup_of"))

    def test_expand_links_in_retrieval(self):
        """检索结果应自动带出关联记忆（式10），否则"追问后续"无从下手"""
        plan = self.mm.save_fta_unit("我打算这周给导师发邮件",
                                     situation_type="homework_or_plan")
        self.mm.save_fta_unit("邮件已经发出去了，还没回复",
                              situation_type="event",
                              link_to=[plan], link_relation="followup_of")
        hits = self.mm.retrieve_fta("给导师发邮件", top_k=3)
        self.assertTrue(hits)
        self.assertIn("linked", hits[0])

    def test_expand_links_can_be_disabled(self):
        self.mm.save_fta_unit("给导师发邮件", situation_type="event")
        hits = self.mm.retrieve_fta("给导师发邮件", top_k=3,
                                    expand_links=False)
        self.assertNotIn("linked", hits[0])


class TestHybridRetrieval(FtaTestBase):
    """式(8)：R = λ·s_emb + (1−λ)·c_struct"""

    def test_lambda_and_tau_match_paper(self):
        """λ=0.8、τ=0.35 是文献5 的取值，改动必须是有意识的"""
        self.assertAlmostEqual(config.MEMORY_RETRIEVAL_LAMBDA, 0.8, places=6)
        self.assertAlmostEqual(config.MEMORY_RETRIEVAL_TAU, 0.35, places=6)

    def test_embedding_backend_is_declared(self):
        """必须如实标注实际用的检索后端（无嵌入模型时是 bigram）

        这一条防的是"看起来是向量检索其实不是"的含混表述 ——
        论文里写"采用向量检索"而实现是字符 n-gram，是实打实的失实。
        """
        self.mm.save_fta_unit("关于考试的焦虑", situation_type="event")
        hits = self.mm.retrieve_fta("考试", top_k=3)
        self.assertTrue(hits)
        self.assertIn(hits[0]["emb_backend"], ("bigram", "chroma"))

    def test_scores_are_sorted_descending(self):
        for text in ("考试压力很大", "考试前一天失眠了", "和室友因为考试复习吵架"):
            self.mm.save_fta_unit(text, situation_type="event")
        hits = self.mm.retrieve_fta("考试", top_k=5)
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_tau_filters_weak_matches(self):
        """τ 必须真的起过滤作用：调高 τ 后命中数不应增加"""
        for text in ("考试压力很大", "今天天气不错", "午饭吃了面"):
            self.mm.save_fta_unit(text, situation_type="event")
        loose = self.mm.retrieve_fta("考试", top_k=10, tau=0.0)
        strict = self.mm.retrieve_fta("考试", top_k=10, tau=0.99)
        self.assertLessEqual(len(strict), len(loose))
        self.assertLess(len(strict), len(loose))

    def test_time_clue_ranks_future_above_past(self):
        """文献5 消融显示**去掉时间线索退化最大**，故时间匹配权重最高

        对"接下来要做什么"这类查询，future 取向的单元必须排在 past 之前。
        """
        self.mm.save_fta_unit("上个月考完的那门课已经出分了",
                              situation_type="event", time_orientation="past",
                              event_time="上个月")
        self.mm.save_fta_unit("下周还有一场考试要考",
                              situation_type="event",
                              time_orientation="future", event_time="下周")
        hits = self.mm.retrieve_fta("下周 考试", top_k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["time_orientation"], "future")

    def test_empty_query_still_returns_recent(self):
        """空查询不得抛错（界面会用它取"最近记得的事"）"""
        self.mm.save_fta_unit("某事", situation_type="event")
        hits = self.mm.retrieve_fta("", top_k=3)
        self.assertIsInstance(hits, list)

    def test_no_units_returns_empty(self):
        self.assertEqual(self.mm.retrieve_fta("任何问题"), [])


class TestForgettingCurve(FtaTestBase):
    """文献4 遗忘曲线 R = e^(−t/S)"""

    def test_monotone_in_time(self):
        self.assertGreater(self.mm.retention(1.0, 0.0),
                           self.mm.retention(1.0, 86400.0))
        self.assertGreater(self.mm.retention(1.0, 86400.0),
                           self.mm.retention(1.0, 30 * 86400.0))

    def test_monotone_in_strength(self):
        """强度越大保留度越高 —— 这是"常用记忆更牢固"的数学形式"""
        self.assertLess(self.mm.retention(1.0, 5 * 86400.0),
                        self.mm.retention(5.0, 5 * 86400.0))

    def test_bounds(self):
        for s in (0.0, 1.0, 10.0):
            for t in (0.0, 86400.0, 1e9):
                r = self.mm.retention(s, t)
                self.assertGreaterEqual(r, 0.0)
                self.assertLessEqual(r, 1.0)

    def test_zero_strength_is_zero(self):
        self.assertEqual(self.mm.retention(0.0, 100.0), 0.0)

    def test_recall_raises_strength_and_resets_clock(self):
        uid = self.mm.save_fta_unit("常被提起的事", situation_type="event")
        before = next(u for u in self.mm.export_user_data()["fta_units"]
                      if u["id"] == uid)
        self.mm.touch_recall([uid])
        after = next(u for u in self.mm.export_user_data()["fta_units"]
                     if u["id"] == uid)
        self.assertAlmostEqual(after["strength"], before["strength"]
                               + config.MEMORY_STRENGTH_GAIN, places=6)
        self.assertGreater(after["last_recall_ts"], before["last_recall_ts"])
        self.assertEqual(after["recall_count"], 1)

    def test_retrieval_itself_touches_recall(self):
        """检索命中即视为一次"复述" —— 否则曲线永远不会变平"""
        uid = self.mm.save_fta_unit("关于期末考试的事", situation_type="event")
        self.mm.retrieve_fta("期末考试", top_k=3)
        unit = next(u for u in self.mm.export_user_data()["fta_units"]
                    if u["id"] == uid)
        self.assertGreaterEqual(unit["recall_count"], 1)

    def test_archive_stale_units_keeps_data(self):
        """归档**不是删除**：数据仍在库里，只是不再参与默认检索

        心理陪伴场景里"用户很久没再提起"本身就是信息，
        删掉会丢失"这件事后来没再出现"这一事实。
        """
        uid = self.mm.save_fta_unit("很久以前的小事", situation_type="event")
        n = self.mm.archive_stale_units(floor=1.1)      # 阈值 >1 → 全部过期
        self.assertGreaterEqual(n, 1)
        unit = next(u for u in self.mm.export_user_data()["fta_units"]
                    if u["id"] == uid)
        self.assertEqual(unit["archived"], 1)
        self.assertEqual(unit["lifecycle"], "archived")
        default_hits = self.mm.retrieve_fta("很久以前的小事", top_k=5)
        self.assertNotIn(uid, [h["id"] for h in default_hits])
        all_hits = self.mm.retrieve_fta("很久以前的小事", top_k=5,
                                        only_active=False)
        self.assertIn(uid, [h["id"] for h in all_hits])


class TestSelfReport(FtaTestBase):
    """主观自评 —— "情绪识别准确率 > 85%" 的真值来源"""

    OBJECTIVE = {"emotion": "高焦虑", "confidence": 0.78, "risk_score": 82,
                 "pad": {"P": -2.1, "A": 1.8, "D": -0.9},
                 "quality": {"jitter": 0.03, "quality_level": "合格"}}

    def test_pair_is_stored_together(self):
        """主观与客观必须**成对**落库

        分开存会让事后对齐变成极易出错的事（时间戳漂移、多设备时钟不同步），
        而准确率指标恰恰依赖这对齐。
        """
        rid = self.mm.save_self_report(8.0, emotion="高焦虑",
                                       note="今天答辩前特别慌",
                                       objective=self.OBJECTIVE)
        self.assertGreater(rid, 0)
        row = self.mm.get_self_reports(limit=1)[0]
        self.assertAlmostEqual(row["score"], 8.0, places=6)
        self.assertEqual(row["emotion"], "高焦虑")
        self.assertEqual(row["o_emotion"], "高焦虑")
        self.assertAlmostEqual(row["o_risk"], 82.0, places=6)
        self.assertAlmostEqual(row["o_pad_p"], -2.1, places=6)
        self.assertEqual(row["o_quality"], "合格")

    def test_consistent_report_is_not_conflict(self):
        """主观 8/10 与客观风险 82 基本一致（差距 2 < 阈值 3）"""
        self.mm.save_self_report(8.0, emotion="高焦虑", objective=self.OBJECTIVE)
        self.assertEqual(self.mm.get_self_reports(1)[0]["conflict"], 0)

    def test_gap_above_threshold_is_conflict(self):
        """主观 2/10（几乎没事）而客观 82（高紧张）→ 必须判为冲突"""
        self.mm.save_self_report(2.0, emotion="平静", objective=self.OBJECTIVE)
        row = self.mm.get_self_reports(1)[0]
        self.assertEqual(row["conflict"], 1)
        self.assertGreaterEqual(row["gap"],
                                config.SELF_REPORT_CONFLICT_GAP * 10)

    def test_label_mismatch_alone_is_conflict(self):
        """分数接近但**标签**不同也算冲突

        这一条是刻意的：分数是粗粒度的一维标量，标签才承载了"是哪种难受"。
        用户说"我是累"而系统判"你是焦虑"，对产品决策影响很大，
        不能因为分数接近就当作一致。
        """
        obj = dict(self.OBJECTIVE, emotion="高焦虑", risk_score=80)
        self.mm.save_self_report(8.0, emotion="疲劳", objective=obj)
        self.assertEqual(self.mm.get_self_reports(1)[0]["conflict"], 1)

    def test_quality_is_carried_for_reliability_counting(self):
        """客观侧质量等级要一起存 —— 只有信号合格的自评才能进准确率分母"""
        self.mm.save_self_report(7.0, objective=self.OBJECTIVE)
        self.mm.save_self_report(7.0, objective=dict(
            self.OBJECTIVE, quality={"jitter": 0.35, "quality_level": "不可用"}))
        stats = self.mm.self_report_consistency()
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["n_reliable"], 1)

    def test_consistency_aggregates(self):
        self.mm.save_self_report(8.0, emotion="高焦虑", objective=self.OBJECTIVE)
        self.mm.save_self_report(1.0, emotion="平静", objective=self.OBJECTIVE)
        stats = self.mm.self_report_consistency()
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["n_conflict"], 1)
        self.assertAlmostEqual(stats["conflict_rate"], 0.5, places=6)
        self.assertGreater(stats["mean_gap"], 0.0)

    def test_consistency_empty_is_safe(self):
        stats = self.mm.self_report_consistency()
        self.assertEqual(stats["n"], 0)
        self.assertEqual(stats["conflict_rate"], 0.0)
        self.assertEqual(stats["mean_gap"], 0.0)

    def test_missing_objective_is_tolerated(self):
        """无客观快照时不得抛错（用户可能先自评、后接传感器）"""
        rid = self.mm.save_self_report(5.0, emotion="低焦虑", objective=None)
        self.assertGreater(rid, 0)
        self.assertEqual(self.mm.get_self_reports(1)[0]["o_emotion"], "")

    def test_ts_can_be_injected(self):
        """时间戳可注入 —— 离线补录历史数据时必须支持"""
        past = time.time() - 86400
        self.mm.save_self_report(6.0, objective=self.OBJECTIVE, ts=past)
        self.assertAlmostEqual(self.mm.get_self_reports(1)[0]["ts"], past,
                               places=3)


class TestDataRights(FtaTestBase):
    """文献5 伦理要求：查看 / 更正 / 删除 / 留存控制"""

    def test_export_contains_self_report(self):
        self.mm.save_self_report(7.0, emotion="低焦虑", objective=None)
        data = self.mm.export_user_data()
        self.assertIn("self_report", data)
        self.assertIn("self_report_consistency", data)
        self.assertEqual(len(data["self_report"]), 1)

    def test_export_contains_fta(self):
        self.mm.save_fta_unit("某事", situation_type="event")
        data = self.mm.export_user_data()
        for key in ("fta_units", "memory_links", "emotion_timeline",
                    "interactions", "profile", "baseline"):
            self.assertIn(key, data)

    def test_purge_clears_self_report(self):
        self.mm.save_self_report(7.0, objective=None)
        self.mm.purge_user_data(keep_profile=True)
        self.assertEqual(self.mm.self_report_consistency()["n"], 0)
        self.assertEqual(len(self.mm.get_self_reports(10)), 0)

    def test_purge_clears_fta(self):
        self.mm.save_fta_unit("某事", situation_type="event")
        self.mm.purge_user_data()
        self.assertEqual(self.mm.retrieve_fta("某事"), [])

    def test_correction_via_update(self):
        """更正权：允许改写事实文本与生命周期"""
        uid = self.mm.save_fta_unit("我打算退学", situation_type="belief_candidate")
        self.mm.update_fta_unit(uid, fact_text="我打算先休学一学期再说",
                                lifecycle="contradicted")
        unit = next(u for u in self.mm.export_user_data()["fta_units"]
                    if u["id"] == uid)
        self.assertIn("休学", unit["fact_text"])
        self.assertEqual(unit["lifecycle"], "contradicted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
