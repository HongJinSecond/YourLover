"""应用程序核心模块"""

import logging
import sys
from typing import Any

from PySide6.QtCore import QRect, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication

from agent.agent_controller import AgentController, AgentMessage
from agent.models.OpenAIModels import DEFAULT_OPENAI_BASE_URL, describe_openai_error
from agent.proactive.engine import ProactiveEngine
from avatar.controller import Live2DController
from core.agent_event import AgentEvent
from core.event_bus import event_bus
from utils.config_manager import config
from widgets.theme import ui_font
from windows.window_manager import WindowManager

logger = logging.getLogger(__name__)


# ============================================================
# Agent 工作线程
# ============================================================


class AgentWorker(QThread):
    """调用 AgentController.chat() 的工作线程。"""

    reply_ready = Signal(object)  # AgentMessage 实例

    def __init__(
        self,
        agent_controller: AgentController,
        user_text: str,
        mini_chat: bool = False,
        configurable: dict | None = None,
        proactive_instruction: str | None = None,
    ) -> None:
        """初始化工作线程。"""
        super().__init__()
        self._agent_controller = agent_controller
        self._user_text = user_text
        self._mini_chat = mini_chat
        # 在线程启动前捕获会话，避免等待期间用户切换窗口改变请求归属。
        self._configurable = dict(
            configurable if configurable is not None else agent_controller.configurable
        )
        self._proactive_instruction = proactive_instruction

    def run(self) -> None:
        """执行 chat() 调用。"""
        proactive = self._proactive_instruction is not None
        if not proactive:  # 主动消息是"她主动发来的"，不显示思考占位
            if self._mini_chat:
                event_bus.mini_chat_thinking_started.emit()
            else:
                event_bus.agent_thinking_started.emit()

        try:
            if proactive:
                result = self._agent_controller.chat_proactive(
                    self._proactive_instruction, self._configurable
                )
            else:
                result = self._agent_controller.chat(self._user_text, self._configurable)
            if not self._mini_chat and not proactive:
                event_bus.agent_thinking_finished.emit()
            self.reply_ready.emit(result)

        except Exception as exc:
            if not self._mini_chat and not proactive:
                event_bus.agent_thinking_finished.emit()
            if self._agent_controller.provider == "openai":
                # 服务商错误正文可能包含敏感信息，只展示受控提示。
                error_text = describe_openai_error(exc)
                logger.warning("OpenAI 兼容接口请求失败: %s", type(exc).__name__)
            else:
                logger.exception("本地模型请求失败")
                error_text = str(exc)
            # 异常时返回错误提示，不让 UI 卡在"思考中"
            self.reply_ready.emit(AgentMessage(f"出错了：{error_text}", "Sad"))

        finally:
            event_bus.avatar_expression_clear.emit()


# ============================================================
# 应用核心类
# ============================================================


class App:
    """桌面助手应用核心，管理整体生命周期。"""

    def __init__(self) -> None:
        """初始化应用核心。"""
        # QApplication 实例（生命周期由 App 持有）
        self._app = QApplication(sys.argv)
        self._app.setQuitOnLastWindowClosed(False)
        self._app.setFont(ui_font())

        # 窗口管理器
        self._window_manager = WindowManager(self._app)

        # 注入 AvatarController（Live2D）
        self._avatar_controller = Live2DController(config)
        self._window_manager.set_avatar_controller(self._avatar_controller)

        # 注册示例 HitArea（根据窗口尺寸划分头部和身体）
        win_w, win_h = config.CHARACTER_WINDOW_SIZE
        self._window_manager.lover_window.register_hit_area("head", QRect(0, 0, win_w, win_h // 6))
        self._window_manager.lover_window.register_hit_area(
            "body", QRect(0, win_h // 6, win_w, win_h - win_h // 6)
        )

        # Agent 相关状态
        self._agent_controller: AgentController | None = None
        # 保存活跃 worker 的集合，防止线程运行期间被 Python GC 销毁
        self._active_workers: set[AgentWorker] = set()
        self._pending_model_reload = False
        self._quit_requested = False
        # AgentController 启动时返回的系统提示（如"连接成功了！"），延迟显示在迷你聊天气泡
        self._startup_message: str | None = None
        # 主动对话引擎（静默检测 → 主动问候/生气），见 docs/proactive_dialogue_mechanism.md
        self._proactive_engine: ProactiveEngine | None = None

        # 初始化 AgentController 并连接 LLM
        self._init_agent_controller()

        # 注册 EventBus 信号路由
        self._setup_event_routing()

    # ============================================================
    # AgentController 初始化
    # ============================================================

    def _init_agent_controller(self) -> None:
        """初始化 AgentController 并连接 LLM。"""
        configurable = (
            dict(self._agent_controller.configurable)
            if self._agent_controller is not None
            else {
                "user_id": "default-user",
                "session_id": "default-session",
            }
        )

        self._agent_controller = AgentController(config, configurable)

        # 两种服务分别保留模型名和地址；老配置缺少 provider 时仍默认 Ollama。
        try:
            provider = getattr(config, "LLM_PROVIDER", "ollama")
            remote = provider == "openai"
            message = self._agent_controller.APIConnect(
                LLM_type=provider,
                LLM_name=getattr(config, "OPENAI_MODEL_ID", "")
                if remote
                else config.CHAT_MODEL_NAME,
                url=getattr(config, "OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
                if remote
                else getattr(config, "api_base", "http://localhost:11434"),
                api_key=getattr(config, "OPENAI_API_KEY", "") if remote else None,
            )
            self._startup_message = getattr(message, "message", None) or str(message)
            logger.info("AgentController 配置完成")
        except Exception as exc:
            self._startup_message = "模型初始化失败，请检查设置。"
            logger.warning("AgentController 初始化失败: %s", type(exc).__name__)

        # 初始化主动对话引擎（连接成功才有效；引擎内部读 PROACTIVE_* 配置）
        self._init_proactive_engine()

    def _init_proactive_engine(self) -> None:
        """创建/重建主动对话引擎。"""
        # 重建前先停掉旧引擎（配置热更新路径会走到这里）
        if self._proactive_engine is not None:
            self._proactive_engine.stop()

        last_activity = None
        try:
            agent = getattr(self._agent_controller, "agent", None)
            if agent is not None:
                last_activity = agent.short_term_memory.get_last_user_activity_time(
                    self._agent_controller.configurable["user_id"]
                )
        except Exception as exc:
            logger.warning("读取上次用户活跃时间失败（从当前时间开始计时）: %s", type(exc).__name__)

        self._proactive_engine = ProactiveEngine(
            config,
            affinity_service=getattr(self._agent_controller, "agent", None)
            and self._agent_controller.agent.affinity_service,
            user_id=self._agent_controller.configurable["user_id"],
            initial_last_activity=last_activity,
        )
        self._proactive_engine.checkInRequested.connect(self._start_proactive_worker)
        self._proactive_engine.angryRequested.connect(self._start_proactive_worker)
        self._proactive_engine.start()

    # ============================================================
    # EventBus 信号路由
    # ============================================================

    def _setup_event_routing(self) -> None:
        """配置 EventBus 信号路由到本类的处理方法。"""
        # 监听用户发送消息，启动 Agent 回复
        event_bus.user_message_sent.connect(self._on_user_message)

        # 监听迷你聊天消息，启动 Agent 回复（回复显示在角色头顶气泡）
        event_bus.mini_chat_input_sent.connect(self._on_mini_chat_message)

        # 监听聊天窗口会话切换，更新 AgentController 的 session_id
        event_bus.chat_session_changed.connect(self._on_chat_session_changed)

        event_bus.settings_batch_changed.connect(self._on_config_update)
        event_bus.application_quit_requested.connect(self.quit)

        # 2D模型管理
        event_bus.avatar_expression_clear.connect(self._avatar_controller.clear_expression)

    # ============================================================
    # 大聊天消息处理
    # ============================================================

    def _on_user_message(self, text: str) -> None:
        """用户发送消息后启动 AgentWorker 调用 AgentController.chat()。"""
        if self._agent_controller is None:
            event_bus.agent_reply_received.emit(
                AgentEvent(type="error", payload={"content": "AgentController 未初始化"})
            )
            return

        # 用户发言 = 活跃信号：重置主动对话状态机（"哄她"机制的入口）
        if self._proactive_engine is not None:
            self._proactive_engine.notify_user_activity()

        worker = AgentWorker(self._agent_controller, text)

        # chat() 完成后，把 AgentMessage 包装成 AgentEvent 发送
        self._start_worker(worker, self._on_agent_reply)

    def _start_worker(self, worker: AgentWorker, reply_handler) -> None:
        """Keep each worker alive until Qt confirms that its thread has finished."""
        if self._quit_requested:
            worker.deleteLater()
            return
        worker.reply_ready.connect(reply_handler)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._active_workers.add(worker)
        worker.start()

    def _on_agent_reply(self, agent_message: AgentMessage) -> None:
        """将 AgentMessage 包装成 AgentEvent，通过 EventBus 发送给聊天窗口。"""
        if self._quit_requested:
            return
        # AgentMessage 有两种构造方式，属性可能不同，做兼容处理
        content = getattr(agent_message, "message", None) or str(agent_message)
        emotion = getattr(agent_message, "ai_emo", "Happy")
        # 好感度：MemoryItem 路径携带轮末最新值；错误/提示消息（字符串构造）为 None
        love = getattr(agent_message, "love", None)

        event = AgentEvent(
            type="reply",
            payload={"content": str(content), "emotion": emotion, "love": love},
        )

        event_bus.agent_reply_received.emit(event)
        event_bus.state_emotion_change.emit(emotion)
        # 仅真实聊天回复携带有效好感度（None 为错误/连接提示，不刷新界面）
        if love is not None:
            event_bus.state_love_change.emit(love)

    def _cleanup_worker(self, worker: AgentWorker) -> None:
        """线程结束后清理引用。"""
        self._active_workers.discard(worker)
        worker.deleteLater()
        if self._quit_requested and not self._active_workers:
            self._window_manager.quit()
        elif self._pending_model_reload and not self._active_workers:
            QTimer.singleShot(0, self._apply_model_settings)

    # ============================================================
    # 主动对话（ProactiveEngine → LLM 异步生成主动消息）
    # ============================================================

    def _start_proactive_worker(self, instruction: str) -> None:
        """用 AgentWorker 异步生成主动消息，复用正常回复的显示链路。"""
        if self._agent_controller is None or not self._agent_controller.is_connected:
            return
        if self._active_workers:  # 正在生成回复时不抢跑
            return

        worker = AgentWorker(
            self._agent_controller,
            "",
            proactive_instruction=instruction,
        )
        self._start_worker(worker, self._on_agent_reply)

    # ============================================================
    # 迷你聊天（快速对话）消息处理
    # ============================================================

    def _on_mini_chat_message(self, text: str) -> None:
        """迷你聊天消息处理：通过 AgentWorker 调用 AgentController。"""
        if self._agent_controller is None:
            event_bus.mini_chat_reply_received.emit("AgentController 未初始化")
            return

        # 迷你聊天发言同样算用户活跃：重置主动对话状态机
        if self._proactive_engine is not None:
            self._proactive_engine.notify_user_activity()

        # 迷你聊天使用固定的 session_id，与大聊天窗口互不影响
        mini_configurable = {
            "user_id": "default-user",
            "session_id": "mini-chat",
        }

        worker = AgentWorker(
            self._agent_controller, text, mini_chat=True, configurable=mini_configurable
        )

        # chat() 完成后，把 AgentMessage 文本通过 mini_chat_reply_received 发送
        self._start_worker(worker, self._on_mini_chat_reply)

    def _on_mini_chat_reply(self, agent_message: AgentMessage) -> None:
        """迷你聊天回复：提取文本，通过 mini_chat_reply_received 发送给头顶气泡。"""
        if self._quit_requested:
            return
        content = getattr(agent_message, "message", None) or str(agent_message)

        event_bus.mini_chat_reply_received.emit(str(content))
        emotion = getattr(agent_message, "ai_emo", "Happy")
        self._avatar_controller.set_state(emotion)
        self._avatar_controller.speak(emotion)

    def _on_chat_session_changed(self, session_id: str) -> None:
        """聊天窗口切换/新建会话时，更新 AgentController 的 session_id。"""
        if self._agent_controller is not None:
            self._agent_controller.configurable["session_id"] = session_id

    # ============================================================
    # 生命周期管理
    # ============================================================

    def run(self) -> int:
        """启动应用：显示窗口并进入 Qt 事件循环。"""
        self._window_manager.run()

        # 延迟显示 AgentController 连接结果（系统提示），确保 UI 已就绪
        # 例如"连接成功了！现在可以开始聊天了！"会自动显示在角色头顶气泡
        if self._startup_message:
            QTimer.singleShot(
                800,
                lambda: event_bus.mini_chat_reply_received.emit(self._startup_message),
            )

        return self._app.exec()

    def quit(self) -> None:
        """Stop scheduling work and let active requests finish before releasing Qt."""
        self._quit_requested = True
        if self._proactive_engine is not None:
            self._proactive_engine.stop()
        for window in (
            self._window_manager.lover_window,
            self._window_manager.chat_window,
            self._window_manager.setting_window,
            self._window_manager.mini_chat_overlay,
        ):
            window.hide_window()
        if not self._active_workers:
            self._window_manager.quit()

    def _on_config_update(self, values: dict[str, Any] | None = None) -> None:
        """保存模型设置后热加载；正在运行的回复完成后再更换控制器。"""
        if values is not None and not (
            values.keys()
            - {"CHAT_DISPLAY_NAME", "CHAT_AVATAR", "PERSONA_PROMPT_MODE", "CUSTOM_PERSONA_PROMPT"}
        ):
            return
        self._pending_model_reload = True
        self._apply_model_settings()

    def _apply_model_settings(self) -> None:
        """安全应用待生效设置，保留当前会话，也不重建无关的 Live2D 控件。"""
        if (
            getattr(self, "_quit_requested", False)
            or not self._pending_model_reload
            or self._active_workers
        ):
            return
        self._pending_model_reload = False
        self._init_agent_controller()

        if self._startup_message:
            QTimer.singleShot(
                800,
                lambda: event_bus.mini_chat_reply_received.emit(self._startup_message),
            )
