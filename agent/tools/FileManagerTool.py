"""Descripttion: 文件管理工具，用于读取和写入文件"""

import os

from agent.tools.BaseTool import BaseTool


class FileManagerTool(BaseTool):
    """文件管理工具，用于读取和写入文件"""

    name = "file_manager"
    description = "用于读取和写入文件。当你需要读取文件内容或保存数据到文件时使用此工具。支持的操作类型：'read'（读取文件）、'write'（写入文件）。"
    param_descriptions = {
        "action": "操作类型，必填，可选值: 'read'（读取文件）或 'write'（写入文件）",
        "file_path": "文件路径，必填，相对路径或绝对路径均可",
        "content": "写入的文件内容，仅当 action='write' 时需要",
    }

    def _tool_call(self, action: str, file_path: str, content: str = "") -> str:
        """执行文件操作"""
        if action == "read":
            if not os.path.exists(file_path):
                return f"错误: 文件 '{file_path}' 不存在"
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif action == "write":
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"文件 '{file_path}' 写入成功"
        else:
            return f"错误: 不支持的操作类型 '{action}'，仅支持 'read' 和 'write'"
