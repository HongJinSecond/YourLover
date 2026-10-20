"""Arithmetic tool exposed to the conversation model."""

from agent.tools.BaseTool import BaseTool


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "用于执行数学计算。当需要计算数学表达式时使用此工具。"
    param_descriptions = {"expression": "数学表达式，例如 '2 + 3 * 4' 或 '5 ** 2'"}

    def _tool_call(self, expression: str) -> str:
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as exc:
            return f"计算错误: {exc}"
