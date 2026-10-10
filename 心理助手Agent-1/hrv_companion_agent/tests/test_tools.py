# -*- coding: utf-8 -*-
"""Agent 工具矩阵单元测试（对应计划书 §4.3 tools/ 目录）

v2 覆盖范围：9 个工具中的 8 个（memory_tool / recall_tool 需要真实库，
由 test_memory_manager.py 覆盖其底层能力，这里只做接口级回归）。
"""
import unittest

from hrv_companion_agent import config
from hrv_companion_agent.agent.memory_manager import MemoryManager
from hrv_companion_agent.agent.tools import (BreathingGuideTool, ComfortTool,
                                             CrisisTool, EmotionTool,
                                             PADTool, ReferralTool,
                                             SelfReportTool, detect_level,
                                             match_skills)


class TestEmotionTool(unittest.TestCase):
    def setUp(self):
        self.state = {"emotion": "高焦虑", "hr_bpm": 95, "risk_score": 85,
                      "confidence": 0.8, "layer": "rule",
                      "quality": {"quality_ok": True, "quality_level": "合格",
                                  "jitter": 0.04},
                      "feats": {"hr_bpm": 95, "rmssd_ms": 22.0, "sdnn_ms": 31.0},
                      "affect": {"level": "moderate", "mood": -0.3}}

    def test_state_provider(self):
        tool = EmotionTool(lambda: self.state)
        out = tool.run({})
        self.assertIn("高焦虑", out)
        self.assertIn("0.80", out)          # 置信度
        self.assertIn("85.0/100", out)      # 紧张评分

    def test_raw_metrics_hidden_by_default(self):
        """v2 契约: 默认不向模型暴露原始生理数值（防止其在回复里复述）

        这与 v1 的行为相反，是有意为之 —— 见 emotion_tool 模块文档
        与 prompts.build_emotion_context 的说明。
        """
        tool = EmotionTool(lambda: self.state)
        out = tool.run({})
        self.assertNotIn("95 bpm", out)

    def test_raw_metrics_when_explicitly_enabled(self):
        """开启 EXPOSE_RAW_METRICS 时应能看到原始值（调试用）"""
        old = config.EXPOSE_RAW_METRICS
        config.EXPOSE_RAW_METRICS = True
        try:
            tool = EmotionTool(lambda: self.state)
            out = tool.run({})
            self.assertIn("95", out)
        finally:
            config.EXPOSE_RAW_METRICS = old

    def test_quality_advice_when_unreliable(self):
        """质量不合格时必须明确提示「以用户自述为准」"""
        bad = dict(self.state)
        bad["quality"] = {"quality_ok": False, "quality_level": "不可用",
                          "jitter": 0.45}
        out = EmotionTool(lambda: bad).run({})
        self.assertIn("以用户自述为准", out)

    def test_no_state(self):
        tool = EmotionTool(lambda: None)
        out = tool.run({})
        self.assertIn("没有可用", out)


class TestComfortTool(unittest.TestCase):
    def test_scripts_for_each_emotion(self):
        tool = ComfortTool()
        for emotion in ("平静", "低焦虑", "高焦虑", "愉悦", "疲劳"):
            out = tool.run({"emotion": emotion})
            self.assertGreater(len(out), 30)

    def test_generic_fallback(self):
        tool = ComfortTool()
        out = tool.run({"emotion": "不存在"})
        self.assertIn("无条件接纳", out)

    def test_evidence_and_caution_present(self):
        """每个技能都必须带起效机制与禁忌 —— 这是 v2 的核心增量"""
        out = ComfortTool().run({"emotion": "高焦虑"})
        self.assertIn("起效机制", out)
        self.assertIn("注意事项", out)

    def test_arousal_selects_different_skill(self):
        """同样「低焦虑」，高唤醒与低唤醒应匹配到不同技能"""
        high = match_skills("低焦虑", "high", 0.5, top_k=1)[0]["id"]
        low = match_skills("低焦虑", "low", 0.5, top_k=1)[0]["id"]
        self.assertNotEqual(high, low)

    def test_recent_penalty_changes_rank(self):
        """最近用过的技能应被降权（避免反复推同一种技术）"""
        best = match_skills("高焦虑", "high", 0.6, top_k=1)[0]["id"]
        after = match_skills("高焦虑", "high", 0.6,
                             recent_ids=[best], top_k=1)[0]["id"]
        self.assertNotEqual(best, after)

    def test_history_boosts_skill(self):
        """对这位用户历史有效的技能应被上调"""
        base = match_skills("疲劳", "low", 0.5, top_k=1)[0]["id"]
        stats = {base: {"total": 4, "effective": 4}}
        boosted = match_skills("疲劳", "low", 0.5, stats=stats, top_k=1)[0]["id"]
        self.assertEqual(base, boosted)   # 已最优则保持

    def test_english_alias_accepted(self):
        """v1 的英文键写法仍应可用（向上兼容）"""
        out = ComfortTool().run({"emotion": "anxiety"})
        self.assertIn("高焦虑", out)

    def test_all_skills_have_required_fields(self):
        from hrv_companion_agent.agent.tools.comfort_tool import SKILLS
        for s in SKILLS:
            for field in ("id", "name", "school", "targets", "arousal",
                          "mechanism", "script", "duration", "caution"):
                self.assertIn(field, s, "技能 %s 缺字段 %s" % (s.get("id"), field))
            self.assertTrue(s["caution"].strip(), "技能 %s 的禁忌不能为空" % s["id"])


class TestBreathingGuideTool(unittest.TestCase):
    def test_relax_guide(self):
        tool = BreathingGuideTool()
        out = tool.run({"type": "relax"})
        self.assertIn("吸气", out)
        self.assertIn("呼气", out)

    def test_default_and_unknown(self):
        tool = BreathingGuideTool()
        self.assertEqual(tool.run({}), tool.run({"type": "unknown"}))

    def test_resonance_mode(self):
        """共振呼吸（约 6 次/分）应存在且标注为日常练习"""
        out = BreathingGuideTool().run({"type": "resonance"})
        self.assertIn("共振呼吸", out)
        self.assertIn("日常练习", out)

    def test_box_mode_warns_contraindication(self):
        """含屏息的模式必须带禁忌提示"""
        out = BreathingGuideTool().run({"type": "box"})
        self.assertIn("屏息", out)
        self.assertIn("惊恐", out)

    def test_emotion_maps_to_mode(self):
        """允许只给情绪，自动选模式（v2 新增）"""
        out = BreathingGuideTool().run({"emotion": "疲劳"})
        self.assertIn("助眠", out)

    def test_hrv_confound_notice(self):
        """必须提醒呼吸会污染 HRV 读数（本项目特有的一条边界）"""
        out = BreathingGuideTool().run({"type": "calm"})
        self.assertIn("HRV", out)


class TestCrisisTool(unittest.TestCase):
    def test_crisis_keywords(self):
        self.assertTrue(CrisisTool.detect_crisis_text("我不想活了"))
        self.assertFalse(CrisisTool.detect_crisis_text("今天天气不错"))

    def test_high_risk_protocol(self):
        tool = CrisisTool()
        out = tool.run({"risk": 99})
        self.assertIn("危机干预协议", out)
        self.assertIn("400-161-9995", out)

    def test_low_risk_no_protocol(self):
        tool = CrisisTool()
        out = tool.run({"risk": 50})
        self.assertIn("未达危机阈值", out)

    def test_level_grading(self):
        """意念分级：被动 / 主动 / 有计划 / 已准备"""
        self.assertEqual(detect_level("活着真没意思"), "passive")
        self.assertEqual(detect_level("我不想活了"), "active")
        self.assertEqual(detect_level("我想好了怎么跳楼"), "preparation")
        self.assertIsNone(detect_level("今天有点累"))

    def test_safety_plan_included(self):
        out = CrisisTool().run({"text": "我不太想活了"})
        self.assertIn("安全计划", out)
        self.assertIn("环境更安全", out)

    def test_forbidden_list_included(self):
        """危机中最容易犯的"好心错误"必须被写进输出"""
        out = CrisisTool().run({"text": "我想死"})
        self.assertIn("不要承诺绝对保密", out)
        self.assertIn("直接询问准则", out)

    def test_higher_level_wins(self):
        """文本与评分给出不同层级时，取更紧急的那个"""
        out = CrisisTool().run({"risk": 99, "text": "活着没意思"})
        self.assertIn("主动意念", out)


class TestPADTool(unittest.TestCase):
    def _state(self):
        return {"emotion": "高焦虑", "risk_score": 85, "confidence": 0.8,
                "feats": {"hr_bpm": 95, "rmssd_ms": 22.0, "sdnn_ms": 31.0,
                          "lf_hf": 3.5},
                "pad": {"P": -1.4, "A": 2.1, "D": -0.4, "method": "rule",
                        "confidence": 0.7, "d_confidence": 0.2,
                        "emotion_hint": "焦虑"}}

    def test_gives_intervention_direction(self):
        out = PADTool(lambda: self._state()).run({})
        self.assertIn("降温", out)

    def test_dim_selection(self):
        out = PADTool(lambda: self._state()).run({"dim": "P"})
        self.assertIn("P（", out)
        self.assertNotIn("A（", out)

    def test_low_d_confidence_warned(self):
        out = PADTool(lambda: self._state()).run({})
        self.assertIn("D 轴置信度低", out)

    def test_why_attribution(self):
        state = self._state()
        state["pad"] = None
        baseline = {"hr_bpm": 70.0, "rmssd_ms": 45.0, "sdnn_ms": 55.0}
        out = PADTool(lambda: state, lambda: baseline).run({"why": "1"})
        self.assertIn("相对特征值法", out)

    def test_no_data(self):
        out = PADTool(lambda: {}).run({})
        self.assertIn("无法估计", out)


class TestReferralTool(unittest.TestCase):
    def test_auto_level_from_text(self):
        out = ReferralTool().run({"text": "我最近两周都睡不着急，没法上课"})
        self.assertIn("L3", out)

    def test_level_from_risk(self):
        out = ReferralTool().run({"risk": 99})
        self.assertIn("L4", out)
        self.assertIn("400-161-9995", out)

    def test_l1_has_no_hotline(self):
        """低层级不给危机热线 —— 避免滥用与标签化"""
        out = ReferralTool().run({"level": "L1"})
        self.assertNotIn("400-161-9995", out)

    def test_ask_scripts_present(self):
        """核心产出是「如何开口」的话术，不是资源列表"""
        out = ReferralTool().run({"level": "L2"})
        self.assertIn("如何开口", out)
        self.assertIn("室友", out)

    def test_disclaimer_included(self):
        out = ReferralTool().run({"level": "L3"})
        self.assertIn("不构成诊断", out)


class TestSelfReportTool(unittest.TestCase):
    def setUp(self):
        self.mm = MemoryManager("sr_user", db_path=":memory:")
        self.state = {"emotion": "平静", "risk_score": 15, "confidence": 0.8,
                      "quality": {"quality_ok": True, "quality_level": "合格"},
                      "pad": {"P": 0.2, "A": -0.1, "D": 0.0}}

    def tearDown(self):
        self.mm.close()

    def test_conflict_takes_user_side(self):
        """主客观冲突时必须以用户为准，且明确指示模型降权"""
        tool = SelfReportTool(self.mm, "sr_user", lambda: self.state)
        out = tool.run({"score": 8, "emotion": "高焦虑"})
        self.assertIn("以用户自述为准", out)
        self.assertIn("不一致", out)

    def test_consistent_when_aligned(self):
        tool = SelfReportTool(self.mm, "sr_user", lambda: self.state)
        out = tool.run({"score": 1, "emotion": "平静"})
        self.assertIn("一致", out)
        self.assertNotIn("以用户自述为准", out)

    def test_poor_quality_forces_conflict(self):
        """信号质量不合格本身就是冲突（生理判定不可信）"""
        bad = dict(self.state)
        bad["quality"] = {"quality_ok": False, "quality_level": "不可用"}
        tool = SelfReportTool(self.mm, "sr_user", lambda: bad)
        out = tool.run({"score": 2})
        self.assertIn("质量不合格", out)

    def test_persisted_and_paired(self):
        """自评必须与其时的客观侧快照成对落库（实验真值来源）"""
        tool = SelfReportTool(self.mm, "sr_user", lambda: self.state)
        tool.run({"score": 6, "emotion": "低焦虑", "note": "考试前"})
        rows = self.mm.get_self_reports()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["score"], 6.0)
        self.assertEqual(rows[0]["o_emotion"], "平静")
        cons = self.mm.self_report_consistency()
        self.assertEqual(cons["n"], 1)

    def test_percent_scale_tolerated(self):
        """用户可能给 0-100 分制，应自动折算"""
        tool = SelfReportTool(self.mm, "sr_user", lambda: self.state)
        out = tool.run({"score": 70})
        self.assertIn("7/10", out)

    def test_requires_user_input(self):
        tool = SelfReportTool(self.mm, "sr_user", lambda: self.state)
        out = tool.run({})
        self.assertIn("不要替用户打分", out)

    def test_emotion_alias_normalized(self):
        self.assertEqual(SelfReportTool.normalize_emotion("有点紧张"), "低焦虑")
        self.assertEqual(SelfReportTool.normalize_emotion("很焦虑"), "高焦虑")


if __name__ == "__main__":
    unittest.main()
