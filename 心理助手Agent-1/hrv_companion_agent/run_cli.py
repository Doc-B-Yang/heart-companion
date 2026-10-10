# -*- coding: utf-8 -*-
"""
命令行对话入口
==============
与"小暖"交互式聊天（流式输出、工具调用、长期记忆）。
自动读取 .env 中的 API Key：有 Key → 真实大模型对话；无 Key/断网 → 离线话术脑。

命令:
  /状态             查看当前情绪状态（生理数据或手动注入）
  /心情 <情绪>      手动注入情绪（平静/愉悦/低焦虑/高焦虑/疲劳），无硬件体验情绪感知
                    示例: /心情 高焦虑 或 /心情 高焦虑 85（自定义风险分）
  /模拟 <场景名>    回放生理场景（stress_event / anxiety_attack /
                    afternoon_fatigue / happy_chat）
  /趋势             查看今日情绪时间线摘要
  /画像             查看系统对你的了解（画像/基线/记忆）
  /帮助             查看本帮助
  /退出             退出
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hrv_companion_agent import config  # noqa: E402
from hrv_companion_agent.core.simulator import (  # noqa: E402
    SIMULATION_SCENARIOS, SensorSimulator)
from hrv_companion_agent.core.system import HeartCompanionSystem  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

HELP_TEXT = """
可用命令:
  /状态                查看当前情绪状态
  /心情 <情绪> [风险]  手动注入情绪（平静/愉悦/低焦虑/高焦虑/疲劳），如 /心情 高焦虑 85
  /模拟 <场景名>       回放生理场景（stress_event / anxiety_attack / afternoon_fatigue / happy_chat）
  /趋势                今日情绪时间线
  /画像                系统对你的了解
  /清除                清除当前注入的情绪状态
  /帮助                查看本帮助
  /退出                退出
直接输入文字即可与小暖聊天。
"""


def print_health(system):
    """打印 LLM 后端与连通性自检结果"""
    status, detail = system.llm.health_check()
    print(f"[系统] LLM后端: {system.llm.backend} | {status}")
    print(f"[系统] {detail}")
    ml = ("已加载随机森林（来源: " + str((system.get_profile().get("ml_source") or "未知"))
          + "）") if system.recognizer.ml.available else "未加载"
    print(f"[系统] 情绪识别: 规则+模糊 在线 | ML层: {ml}")


def run_scenario(system, name: str):
    if name not in SIMULATION_SCENARIOS:
        print(f"未知场景: {name}，可选: {list(SIMULATION_SCENARIOS)}")
        return
    sim = SensorSimulator(scenario=name, seed=42)
    print(f"▶ 回放场景「{name}」（模拟 ESP32 生理数据）……")
    last = None
    for i, pkt in enumerate(sim.run_all(), 1):
        result = system.tick(pkt)
        if result is None:
            continue
        state = result["state"]
        if state["emotion"] != last or i == 1:
            print(f"  [{i:>3}s] 识别情绪: {state['emotion']} | "
                  f"风险 {state['risk_score']:.0f}/100")
            last = state["emotion"]
        if result.get("should_trigger"):
            print(f"\n  🔔 主动触发（{result['trigger_level']}）: {result['message']}\n")
    print("  ✅ 场景回放完成（情绪时间线已存入长期记忆）")


def welcome(system):
    name = config.COMPANION_NAME
    profile = system.memory.get_user_profile()
    print(f"\n  🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷")
    print(f"   你好呀，我是{name}，你的情感陪伴助手。")
    if profile.get("facts"):
        recent_fact = list(profile["facts"].values())[-1]
        print(f"   我记得你哦：{recent_fact.split('（')[0]}")
    else:
        print("   我会感知你的心率与情绪，在你需要的时候主动出现。")
    print("   输入 /帮助 查看命令，直接打字就可以和我聊天。")
    print("  🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷🌷\n")


def main():
    system = HeartCompanionSystem(user_id=config.DEFAULT_USER_ID)
    print_health(system)
    welcome(system)

    while True:
        try:
            text = input("你 👤: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见，照顾好自己。🌷")
            break
        if not text:
            continue

        # ---------------- 命令处理 ----------------
        if text in ("/退出", "/exit", "exit", "quit"):
            print(f"{config.COMPANION_NAME}: 再见啦，我会记得我们的聊天。"
                  "需要的时候随时来找我。🌷")
            break
        if text in ("/帮助", "/help", "help"):
            print(HELP_TEXT)
            continue
        if text == "/状态":
            state = system.get_state()
            if state:
                print(f"当前情绪: {state.get('emotion')} | "
                      f"风险 {state.get('risk_score', 0):.0f}/100 | "
                      f"来源: {state.get('layer', 'N/A')}")
            else:
                print("还没有情绪数据。可以 /模拟 stress_event 或 /心情 高焦虑")
            continue
        if text.startswith("/心情"):
            parts = text.split()
            emotion = parts[1] if len(parts) >= 2 else ""
            risk = parts[2] if len(parts) >= 3 else None
            if not emotion:
                print("用法: /心情 高焦虑 [风险分0-100]，"
                      "可选: 平静/愉悦/低焦虑/高焦虑/疲劳")
                continue
            result = system.set_manual_emotion(emotion, risk)
            print(result.get("hint") or result.get("error"))
            continue
        if text == "/清除":
            system._state = None           # noqa: SLF001
            system.agent.set_state(None)
            print("已清除情绪状态，恢复普通聊天模式。")
            continue
        if text.startswith("/模拟"):
            run_scenario(system, text.split()[-1] if len(text.split()) > 1
                         else "stress_event")
            continue
        if text == "/趋势":
            timeline = system.get_timeline(days=1)
            if not timeline:
                print("今天还没有情绪记录。")
            else:
                print("今日情绪时间线: " + " → ".join(
                    t["emotion"] for t in timeline[-40:]))
            continue
        if text == "/画像":
            prof = system.get_profile()
            print("系统对你的了解:")
            print(f"  画像: {prof['profile']}")
            print(f"  生理基线: {prof['baseline']}")
            print(f"  记忆统计: {prof['stats']}")
            continue

        # ---------------- 正常对话 ----------------
        print(f"{config.COMPANION_NAME} 🌷: ", end="", flush=True)
        for chunk in system.stream_chat(text):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
