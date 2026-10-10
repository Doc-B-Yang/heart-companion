# -*- coding: utf-8 -*-
"""
工具基类
========
对齐 hello_agents.tools.Tool 的核心约定（name / description / run(params)），
使本项目的四大工具既可以挂载到 CompanionAgent（内置 ToolRegistry），
也可以直接注册进 hello_agents.ToolRegistry 复用。
"""

from __future__ import annotations

from typing import Dict


class BaseTool:
    """Agent 工具基类"""

    name: str = "base_tool"
    description: str = ""

    def run(self, params: Dict) -> str:
        """执行工具，返回文本结果（注入对话上下文）"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"


class SimpleToolRegistry:
    """
    内置轻量工具注册表（接口与 hello_agents.ToolRegistry 兼容）:
      register_tool / get_tool / get_tools_description / list_tools / execute_tool
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        self._tools.pop(name, None)

    def get_tool(self, name: str):
        return self._tools.get(name)

    def list_tools(self) -> list:
        return list(self._tools.keys())

    def execute_tool(self, name: str, *args) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"❌ 错误：未找到工具 '{name}'"
        if args and isinstance(args[0], str):
            return tool.run({"input": args[0]})
        return tool.run(args[0] if args else {})

    def get_tools_description(self) -> str:
        if not self._tools:
            return "暂无可用工具"
        lines = []
        for t in self._tools.values():
            lines.append(f"- **{t.name}**: {t.description}")
        return "\n".join(lines)
