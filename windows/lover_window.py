"""桌面角色窗口模块"""

from collections import deque

from PySide6.QtCore import QEvent, QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QWidget

from core.event_bus import event_bus
from utils.config_manager import config
from widgets.menu.context_menu import ContextMenu
from windows.base_window import BaseWindow


class LoverWindow(BaseWindow):
    """虚拟伴侣悬浮窗口。"""

    hit_area_clicked = Signal(str)

    # 窗口位置变化信号，供 MiniChatOverlay 等跟随
    position_changed = Signal(QPoint)

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化悬浮窗口。"""
        self._avatar_widget: QWidget | None = None
        self._hit_areas: dict[str, QRect] = {}
        self._context_menu: ContextMenu | None = None
        self._draggable: bool = True
        self._dragging: bool = False
        self._drag_start_pos: QPoint = QPoint()
        self._drag_start_global_pos: QPoint = QPoint()

        # 拖拽动画控制：只有真正发生定向移动后才触发拖拽动画
        self._drag_started_emitted: bool = False  # 是否已发射 avatar_drag_started
        self._drag_move_threshold: int = 20  # 触发拖拽动画的最小累计移动距离（像素）

        # 摇晃检测：记录最近 N 次水平拖拽位移
        self._drag_history: deque[int] = deque(maxlen=20)
        self._shake_threshold: int = 3  # 方向变化次数阈值
        self._shake_cooldown: bool = False
        self._shake_triggered: bool = False  # 本次按下-释放过程中是否触发过摇晃
        super().__init__(parent)

    def _setup_window_flags(self) -> None:
        """配置悬浮窗口标志：无边框、透明、置顶、不在任务栏显示。"""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def setup_ui(self) -> None:
        """创建窗口基础布局。"""
        # 中央占位控件，等待 Avatar 渲染控件注入
        self._central_widget = QWidget(self)
        self._central_widget.setObjectName("lover_central")
        self.setCentralWidget(self._central_widget)

        # 默认尺寸，后续可根据 Avatar 模型尺寸调整
        self.setFixedSize(*config.CHARACTER_WINDOW_SIZE)

        # 默认右键菜单
        self._context_menu = ContextMenu(self)

        self.apply_style()

    def connect_signals(self) -> None:
        """连接事件总线与本窗口相关信号。"""
        self.hit_area_clicked.connect(event_bus.avatar_hit_area_clicked.emit)

    def apply_style(self) -> None:
        """应用悬浮窗透明样式。"""
        self.setStyleSheet(
            """
            #lover_central {
                background-color: transparent;
            }
            """
        )

    # ============================================================
    # Avatar 与 HitArea 接口
    # ============================================================

    def set_avatar_widget(self, widget: QWidget) -> None:
        """注入 Avatar 渲染控件。"""
        # 移除旧 widget 的事件过滤器
        if self._avatar_widget is not None:
            self._avatar_widget.removeEventFilter(self)
            self._avatar_widget.setParent(None)
            self._avatar_widget.deleteLater()

        self._avatar_widget = widget
        self.setCentralWidget(widget)

        # ★ 关键：让 Avatar widget 鼠标事件穿透到本窗口
        # 这样 mousePressEvent / mouseMoveEvent / mouseReleaseEvent 能正常触发
        widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # 后备：安装事件过滤器（穿透生效时不会被触发，无副作用）
        widget.installEventFilter(self)

    def register_hit_area(self, name: str, rect: QRect) -> None:
        """注册一个可点击区域（HitArea）。"""
        self._hit_areas[name] = rect

    def unregister_hit_area(self, name: str) -> None:
        """注销可点击区域。"""
        self._hit_areas.pop(name, None)

    def clear_hit_areas(self) -> None:
        """清空所有 HitArea。"""
        self._hit_areas.clear()

    def on_avatar_clicked(self, pos: QPoint) -> None:
        """处理 Avatar 区域内的点击。"""
        for name, rect in self._hit_areas.items():
            if rect.contains(pos):
                self.hit_area_clicked.emit(name)
                return

    # ============================================================
    # 事件过滤器（拦截 Avatar widget 的鼠标事件）
    # ============================================================

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        """事件过滤器：拦截 Avatar widget 的鼠标事件。"""
        if obj is self._avatar_widget:
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress:
                self._handle_mouse_press(event)
                return True
            elif etype == QEvent.Type.MouseMove:
                self._handle_mouse_move(event)
                return True
            elif etype == QEvent.Type.MouseButtonRelease:
                self._handle_mouse_release(event)
                return True
        return super().eventFilter(obj, event)

    def _map_to_window(self, pos: QPoint) -> QPoint:
        """将 Avatar widget 本地坐标转换为 LoverWindow 本地坐标。"""
        if self._avatar_widget is not None:
            return self._avatar_widget.mapTo(self, pos)
        return pos

    # ============================================================
    # 窗口位置跟踪
    # ============================================================

    def moveEvent(self, event) -> None:  # noqa: N802
        """窗口移动时发射 position_changed 信号。"""
        super().moveEvent(event)
        self.position_changed.emit(self.pos())

    # ============================================================
    # 拖拽交互
    # ============================================================

    def set_draggable(self, enabled: bool) -> None:
        """启用或禁用窗口拖拽。"""
        self._draggable = enabled

    def _handle_mouse_press(self, event: QMouseEvent) -> None:
        """处理鼠标按下（由 mousePressEvent 或事件过滤器调用）。"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            # 坐标统一转换为 LoverWindow 本地坐标
            self._drag_start_pos = self._map_to_window(event.position().toPoint())
            self._drag_start_global_pos = event.globalPosition().toPoint()
            # 重置拖拽动画标志：等真正移动后才触发
            self._drag_started_emitted = False
            # 重置摇晃标志：本次按下-释放周期开始
            self._shake_triggered = False
        elif event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())

    def _handle_mouse_move(self, event: QMouseEvent) -> None:
        """处理鼠标移动：执行窗口拖拽并检测摇晃手势。"""
        if not self._draggable or not self._dragging:
            return

        # 本次移动的增量（_drag_start_global_pos 每帧更新，所以 delta 是单帧增量）
        delta = event.globalPosition().toPoint() - self._drag_start_global_pos

        # 首次定向移动才触发拖拽动画
        if not self._drag_started_emitted:
            move_distance = delta.manhattanLength()
            if move_distance >= self._drag_move_threshold:
                # 累计移动超过阈值，确认是拖拽而非点击
                self._drag_started_emitted = True
                direction = self._calc_drag_direction(delta.x(), delta.y())
                # 拖拽开始，携带初始方向
                event_bus.avatar_drag_started.emit(direction)

        # 执行窗口移动
        self.move(self.pos() + delta)
        self._drag_start_global_pos = event.globalPosition().toPoint()

        # 摇晃检测：记录水平位移方向变化
        self._drag_history.append(delta.x())
        self._detect_shake()

    @staticmethod
    def _calc_drag_direction(dx: int, dy: int) -> str:
        """根据单帧移动增量计算主方向。"""
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

    def _handle_mouse_release(self, event: QMouseEvent) -> None:
        """处理鼠标释放：根据本次按下-释放过程的行为，触发对应事件。"""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False

            if self._shake_triggered:
                # 摇晃过：摇晃动画已播放，释放时不触发任何事件
                return
            elif self._drag_started_emitted:
                # 拖拽过：触发拖拽结束动画
                event_bus.avatar_drag_finished.emit(
                    event.globalPosition().toPoint().x(),
                    event.globalPosition().toPoint().y(),
                )
            else:
                # 未发生拖拽/摇晃，判断是否为快速点击
                local_pos = self._map_to_window(event.position().toPoint())
                move_distance = (local_pos - self._drag_start_pos).manhattanLength()
                if move_distance < 5:
                    self.on_avatar_clicked(local_pos)
                # else: 轻微移动但未达拖拽阈值，忽略

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """鼠标按下事件（点击在窗口空白区域时触发，Avatar 区域由事件过滤器处理）。"""
        self._handle_mouse_press(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """鼠标移动事件。"""
        self._handle_mouse_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """鼠标释放事件。"""
        self._handle_mouse_release(event)

    # ============================================================
    # 右键菜单
    # ============================================================

    def _show_context_menu(self, pos: QPoint) -> None:
        """在指定位置显示右键菜单。"""
        if self._context_menu is not None:
            self._context_menu.popup(pos)

    # ============================================================
    # 动画与效果接口（委托给 AvatarController）
    # ============================================================

    def _detect_shake(self) -> None:
        """基于拖拽历史检测摇晃手势。"""
        if self._shake_cooldown or len(self._drag_history) < 4:
            return

        direction_changes = 0
        last_direction = 0
        for dx in self._drag_history:
            if dx == 0:
                continue
            direction = 1 if dx > 0 else -1
            if last_direction != 0 and direction != last_direction:
                direction_changes += 1
            last_direction = direction

        if direction_changes >= self._shake_threshold:
            self.shake_avatar()
            self._shake_triggered = True  # 标记本次发生过摇晃，释放时不触发点击/拖拽结束
            self._shake_cooldown = True
            self._drag_history.clear()
            QTimer.singleShot(1000, self._reset_shake_cooldown)

    def _reset_shake_cooldown(self) -> None:
        """重置摇晃检测冷却状态。"""
        self._shake_cooldown = False

    def shake_avatar(self) -> None:
        """通过事件总线触发已接入的角色摇晃动画。"""
        event_bus.avatar_shaken.emit()

    def set_avatar_state(self, state: str) -> None:
        """设置 Avatar 状态。"""
        event_bus.avatar_state_changed.emit(state)
