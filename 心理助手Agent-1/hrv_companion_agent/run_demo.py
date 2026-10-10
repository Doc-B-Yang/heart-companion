# -*- coding: utf-8 -*-
"""
端到端闭环演示
==============
无需任何硬件与 API Key 即可运行，完整演示:
  生理感知(模拟BLE数据包) → HRV特征提取 → 三层情绪识别
  → 主动触发决策 → 小暖主动关怀 → 反馈自适应优化 → 长期记忆沉淀

用法:
  python run_demo.py                    # 演示"考试压力事件"场景
  python run_demo.py --scenario anxiety_attack
  python run_demo.py --no-chat          # 演示后不进入交互聊天
"""

import argparse
import sys
import time
from pathlib import Path

# 路径引导: 使 hrv_companion_agent 包可导入（无论从何处启动）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hrv_companion_agent import config  # noqa: E402
from hrv_companion_agent.core.simulator import (  # noqa: E402
    SIMULATION_SCENARIOS, SensorSimulator)
from hrv_companion_agent.core.system import HeartCompanionSystem  # noqa: E402

# Windows 控制台 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

BANNER = r"""
  ╔══════════════════════════════════════════════════════════╗
  ║   主动式情感陪伴系统 —— 端到端闭环演示                      ║
  ║   基于实时心率变异性(HRV)驱动 · 北京交通大学大创项目         ║
  ╚══════════════════════════════════════════════════════════╝"""

EMOTION_ICON = {"平静": "🟢", "愉悦": "😊", "低焦虑": "🟡", "高焦虑": "🔴", "疲劳": "😴"}
EMOTION_CN = {"平静": "平静", "愉悦": "愉悦", "低焦虑": "低焦虑",
              "高焦虑": "高焦虑", "疲劳": "疲劳"}


def main():
    parser = argparse.ArgumentParser(description="端到端闭环演示")
    parser.add_argument("--scenario", default="stress_event",
                        choices=list(SIMULATION_SCENARIOS))
    parser.add_argument("--no-chat", action="store_true",
                        help="演示后不进入交互聊天")
    parser.add_argument("--user", default=config.DEFAULT_USER_ID)
    args = parser.parse_args()

    print(BANNER)
    print(f"\n📋 场景: {args.scenario}（模拟 ESP32 每 1 秒发送 1 包 BLE 数据）")
    print("   说明: 场景中情绪为「剧本标注」，系统只能看到生理数据，情绪靠算法识别")

    # 初始化闭环系统
    system = HeartCompanionSystem(user_id=args.user)
    profile = system.get_profile()
    status, detail = system.llm.health_check()
    print(f"\n🧠 LLM 后端: {profile['llm_backend']} | {status}")
    print(f"   {detail}")
    print(f"🤖 ML 分类器(Layer3): {'已加载随机森林模型（来源: ' + str(profile.get('ml_source') or '未知') + '）' if profile['ml_loaded'] else '未加载（自动跳过）'}")

    # 场景回放
    sim = SensorSimulator(scenario=args.scenario)
    print("\n" + "─" * 64)
    print("▶ 开始接收生理数据（1 包/秒）……")
    print("─" * 64)

    last_emotion, last_risk = None, None
    trigger_count = 0
    pending_feedback = None  # (trigger_risk, trigger_tick)
    tick = 0

    for packet in sim.run_all():
        tick += 1
        ts = time.time()
        result = system.tick(packet)
        state = result["state"]

        # 情绪状态变化时打印一行
        if state["emotion"] != last_emotion or abs(state["risk_score"] - (last_risk or 0)) > 8:
            icon = EMOTION_ICON.get(state["emotion"], "❓")
            print(f"[{tick:>3}s] {icon} 识别情绪: {state['emotion']:<4} "
                  f"| 风险 {state['risk_score']:>5.1f}/100 "
                  f"| HR {state['hr_bpm']:.0f} bpm | 层: {state['layer']}")
            last_emotion, last_risk = state["emotion"], state["risk_score"]
        else:
            print(".", end="", flush=True)

        # 主动触发 → 小暖出场
        if result.get("should_trigger"):
            trigger_count += 1
            print(f"\n\n🔔 ===== 主动触发（{result['trigger_level']} | "
                  f"策略: {result['strategy']['name']}）=====")
            print(f"    风险评分 {result['risk']:.1f}/100，原因: {result['reason']}")
            print(f"🌷 小暖: {result['message']}")
            if result.get("tts_audio"):
                print(f"🔊 TTS 语音已合成: {result['tts_audio']}")
            print("=" * 64)
            pending_feedback = (result["risk"], tick)

        # 干预后 10 秒评估效果（模拟生理反馈闭环 STEP 06）
        if pending_feedback and tick - pending_feedback[1] >= 10:
            before_risk, _ = pending_feedback
            after_risk = state["risk_score"]
            fb = system.feedback(after_risk=after_risk)
            if fb.get("recorded"):
                print(f"\n📈 反馈闭环: 干预前风险 {before_risk:.0f} → 干预后 {after_risk:.0f}"
                      f"（{'✅ 有效' if fb['effective'] else '❌ 未达预期'}）")
                print(f"   个体触发阈值偏移: {fb['trigger_threshold_offset']:+.1f}"
                      "（有效→阈值上调减少打扰 / 无效→阈值下调更早介入）")
            pending_feedback = None

        # 模拟真实时间（1 秒 1 包）
        time.sleep(0.05)  # 演示提速 20 倍；真实硬件为 1.0s

    print("\n\n" + "─" * 64)
    print("✅ 场景回放完成。以下是系统沉淀的长期记忆:")
    print("─" * 64)

    timeline = system.get_timeline(days=1)
    emo_seq = [t["emotion"] for t in timeline]
    print(f"📊 情绪时间线（{len(timeline)} 条记录）:")
    print("   " + " → ".join(EMOTION_CN.get(e, e) for e in emo_seq[:60]) +
          (" …" if len(emo_seq) > 60 else ""))
    stats = system.memory.stats()
    print(f"💾 记忆统计: 互动 {stats['interactions']} 条 | 情绪记录 {stats['emotion_records']} 条")
    baseline = stats.get("baseline") or {}
    if baseline:
        print(f"📏 个体基线(EMA): " + ", ".join(
            f"{k}={v}" for k, v in baseline.items() if k != "updated_at"))
    print(f"🎯 主动触发次数: {trigger_count}")

    # 语义记忆检索演示
    hits = system.memory.semantic_search("焦虑 紧张")
    if hits:
        print(f"🔍 语义检索「焦虑 紧张」→ 命中 {len(hits)} 条历史互动（示例: "
              f"{hits[0]['content'][:40]}…）")

    if not args.no_chat:
        print("\n" + "─" * 64)
        print("💬 现在可以和小暖聊天（输入 exit 退出）")
        print("─" * 64)
        while True:
            msg = input("\n你 👤: ").strip()
            if msg.lower() in ("exit", "quit", "退出", "q"):
                break
            if not msg:
                continue
            reply = system.chat(msg)
            print(f"\n🌷 小暖: {reply['response']}")

    print("\n🎉 演示结束。完整文档见 说明书.md / README.md")
    system.close()


if __name__ == "__main__":
    main()
