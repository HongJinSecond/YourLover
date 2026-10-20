"""Discover concrete tools and expose their signatures to LangChain."""

import importlib
import inspect
import logging
import pkgutil
from abc import ABC, abstractmethod
from typing import Annotated, Any, get_type_hints

from langchain_core.tools import StructuredTool, tool

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    param_descriptions: dict[str, str] = {}

    @abstractmethod
    def _tool_call(self, **kwargs) -> Any:
        """Implement the tool's operation with explicitly typed parameters."""
        raise NotImplementedError

    def tool_call(self, **kwargs) -> Any:
        return self._tool_call(**kwargs)


def _wrap_tool_instance(instance: BaseTool) -> StructuredTool:
    hints = get_type_hints(instance._tool_call)
    signature = inspect.signature(instance._tool_call)
    parameters = [
        parameter.replace(
            annotation=Annotated[hints.get(name, str), instance.param_descriptions.get(name, "")]
        )
        for name, parameter in signature.parameters.items()
        if parameter.kind != inspect.Parameter.VAR_KEYWORD
    ]

    def invoke(**kwargs):
        return instance.tool_call(**kwargs)

    # LangChain uses inspect.signature and type hints to generate the JSON schema.
    # Preserve defaults and descriptions directly instead of generating Python source.
    invoke.__name__ = instance.name
    invoke.__doc__ = instance.description
    invoke.__signature__ = signature.replace(parameters=parameters)
    invoke.__annotations__ = {parameter.name: parameter.annotation for parameter in parameters}
    if "return" in hints:
        invoke.__annotations__["return"] = hints["return"]
    return tool(description=instance.description)(invoke)


def discover_and_wrap_tools(package_name: str = "agent.tools") -> dict[str, StructuredTool]:
    tools: dict[str, StructuredTool] = {}
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        logger.exception("无法导入工具包 %s", package_name)
        return tools

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        if module_name == "BaseTool":
            continue
        full_name = f"{package_name}.{module_name}"
        try:
            module = importlib.import_module(full_name)
        except Exception:
            logger.exception("无法加载工具模块 %s", full_name)
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not (
                issubclass(cls, BaseTool)
                and cls is not BaseTool
                and not inspect.isabstract(cls)
                and cls.__module__ == full_name
            ):
                continue
            if not cls.name or cls.name in tools:
                logger.warning("跳过工具 %s：名称为空或重复", cls.__name__)
                continue
            try:
                tools[cls.name] = _wrap_tool_instance(cls())
            except Exception:
                logger.exception("无法注册工具 %s", cls.name)
    return tools
