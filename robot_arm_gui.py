from __future__ import annotations

import sys
import time
import threading
from dataclasses import dataclass
from typing import Dict

import serial
from serial.tools import list_ports

try:
    from PyQt6.QtCore import QSignalBlocker, QTimer, Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QScrollArea,
        QSlider,
        QSpinBox,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit("PyQt6 未安裝，請先執行 `pip install PyQt6`。") from exc

import cubemotorAK109Util as motor_proto

APP_TITLE = "Robotic Arm Safe Console"
CAN_BAUDRATE = 2_000_000
SERIAL_TIMEOUT = 0.05
WRITE_TIMEOUT = 0.20
STATUS_POLL_MS = 150
COMMAND_RATE_LIMIT_S = 0.08
SAFE_HOLD_SPEED_ERPM = 3_000
SAFE_HOLD_RPA = 10_000
DEFAULT_TEMP_LIMIT_C = 70
SLIDER_SCALE = 10
BUS_KEYS = ("A",)


@dataclass
class MotorSpec:
    key: str
    name: str
    family: str
    bus_key: str
    motor_id: int
    min_deg: float
    max_deg: float
    speed_erpm: int
    rpa: int
    accent: str


@dataclass
class MotorState:
    target: float = 0.0
    position: float = 0.0
    speed: float = 0.0
    current: float = 0.0
    temperature: float = 0.0
    error: int = 0
    bus_connected: bool = False
    telemetry_ok: bool = False
    target_initialized: bool = False
    last_update: float = 0.0


DEFAULT_MOTORS = [
    MotorSpec("joint_a", "Joint A", "AK10-9", "A", 0, -45.0, 45.0, 6_000, 20_000, "#35c2a1"),
    MotorSpec("joint_b", "Joint B", "AK10-9", "A", 1, -45.0, 45.0, 6_000, 20_000, "#ff8f3d"),
    MotorSpec("joint_c", "Joint C", "AK70", "A", 93, -30.0, 30.0, 4_000, 12_000, "#5bbcff"),
]

DEFAULT_PORTS = {"A": "COM18"}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def build_bridge_init_frame() -> bytearray:
    frame = [
        0xAA, 0x55, 0x12, 0x01, 0x02,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    ]
    checksum = sum(frame[2:19]) & 0xFF
    frame.append(checksum)
    return bytearray(frame)


class CanBusController:
    def __init__(self, bus_key: str):
        self.bus_key = bus_key
        self.port = ""
        self.ser: serial.Serial | None = None
        self.listener: motor_proto.SerialCanListener | None = None
        self.motor_ids: set[int] = set()
        self._lock = threading.Lock()
        self._last_command_ts: Dict[int, float] = {}
        self._states: Dict[int, dict] = {}

    @property
    def connected(self) -> bool:
        return bool(self.ser and self.ser.is_open and self.listener)

    def connect(self, port: str, motor_ids: list[int]) -> None:
        self.disconnect()
        ser: serial.Serial | None = None
        try:
            ser = serial.Serial(
                port,
                CAN_BAUDRATE,
                timeout=SERIAL_TIMEOUT,
                write_timeout=WRITE_TIMEOUT,
            )
            time.sleep(0.1)
            ser.write(build_bridge_init_frame())
            time.sleep(0.1)
            listener = motor_proto.SerialCanListener(ser)
            listener._th.start()
        except Exception:
            if ser and ser.is_open:
                ser.close()
            raise

        self.port = port
        self.ser = ser
        self.listener = listener
        self.motor_ids = set(motor_ids)
        self._last_command_ts.clear()
        self._states.clear()

    def disconnect(self) -> None:
        with self._lock:
            listener = self.listener
            ser = self.ser
            self.listener = None
            self.ser = None
            self.motor_ids = set()
            self._states.clear()
            self._last_command_ts.clear()

        if listener:
            try:
                listener.close()
            except Exception:
                pass

        if ser and ser.is_open:
            try:
                ser.close()
            except Exception:
                pass

    def set_zero(self, motor_id: int) -> None:
        with self._lock:
            self._require_connection()
            motor_proto.servo_mod_set_zero(self.ser, control_mode_id=5, motor_id=motor_id)

    def send_position(
        self,
        motor_id: int,
        target_deg: float,
        speed_erpm: int,
        rpa: int,
        force: bool = False,
    ) -> bool:
        with self._lock:
            self._require_connection()
            now = time.time()
            if not force and now - self._last_command_ts.get(motor_id, 0.0) < COMMAND_RATE_LIMIT_S:
                return False
            motor_proto.servo_mod_pos_speed(
                self.ser,
                control_mode_id=6,
                motor_id=motor_id,
                pos_deg=target_deg,
                speed_erpm=max(1, int(speed_erpm)),
                rpa=max(1, int(rpa)),
            )
            self._last_command_ts[motor_id] = now
            return True

    def hold_position(self, motor_id: int, fallback: float = 0.0) -> None:
        snapshot = self.snapshot_states()
        current_position = snapshot.get(motor_id, {}).get("position", fallback)
        self.send_position(
            motor_id,
            target_deg=current_position,
            speed_erpm=SAFE_HOLD_SPEED_ERPM,
            rpa=SAFE_HOLD_RPA,
            force=True,
        )

    def hold_all(self, fallbacks: Dict[int, float] | None = None) -> None:
        fallbacks = fallbacks or {}
        for motor_id in list(self.motor_ids):
            try:
                self.hold_position(motor_id, fallback=fallbacks.get(motor_id, 0.0))
            except Exception:
                continue

    def snapshot_states(self) -> Dict[int, dict]:
        listener = self.listener
        if not listener:
            return {}
        raw = listener.get_status()
        now = time.time()
        for motor_id, payload in raw.items():
            if motor_id not in self.motor_ids:
                continue
            self._states[motor_id] = {
                "position": float(payload["position"]),
                "speed": float(payload["speed"]),
                "current": float(payload["current"]),
                "temperature": float(payload["temperature"]),
                "error": int(payload["error"]),
                "last_update": now,
            }
        return dict(self._states)

    def _require_connection(self) -> None:
        if not self.connected or not self.ser:
            raise RuntimeError(f"Bus {self.bus_key} 尚未連線。")


class ArmController:
    def __init__(self, specs: list[MotorSpec]):
        self.specs: Dict[str, MotorSpec] = {spec.key: spec for spec in specs}
        self.states: Dict[str, MotorState] = {
            spec.key: MotorState(target=0.0) for spec in specs
        }
        self.buses = {bus_key: CanBusController(bus_key) for bus_key in BUS_KEYS}
        self.motion_armed = False

    def update_spec(self, spec: MotorSpec) -> None:
        self.specs[spec.key] = spec
        state = self.states[spec.key]
        state.target = clamp(state.target, spec.min_deg, spec.max_deg)

    def set_motion_armed(self, armed: bool) -> None:
        self.motion_armed = armed

    def connect_buses(self, port_map: Dict[str, str]) -> None:
        assignments: Dict[str, list[int]] = {bus_key: [] for bus_key in BUS_KEYS}
        for spec in self.specs.values():
            assignments[spec.bus_key].append(spec.motor_id)

        for bus_key, motor_ids in assignments.items():
            if len(motor_ids) != len(set(motor_ids)):
                raise RuntimeError(f"Bus {bus_key} 有重複的 motor ID，請先修正。")

        for bus_key, motor_ids in assignments.items():
            controller = self.buses[bus_key]
            if not motor_ids:
                controller.disconnect()
                continue
            port = port_map.get(bus_key, "").strip()
            if not port:
                raise RuntimeError(f"Bus {bus_key} 沒有指定 COM port。")
            controller.connect(port, motor_ids)

        for state in self.states.values():
            state.target_initialized = False
            state.telemetry_ok = False

    def disconnect_all(self) -> None:
        for controller in self.buses.values():
            controller.disconnect()
        for state in self.states.values():
            state.bus_connected = False
            state.telemetry_ok = False

    def refresh_states(self) -> Dict[str, MotorState]:
        snapshots = {
            bus_key: controller.snapshot_states()
            for bus_key, controller in self.buses.items()
        }
        for spec in self.specs.values():
            controller = self.buses[spec.bus_key]
            snapshot = snapshots[spec.bus_key]
            payload = snapshot.get(spec.motor_id)
            state = self.states[spec.key]
            state.bus_connected = controller.connected
            if payload:
                state.position = payload["position"]
                state.speed = payload["speed"]
                state.current = payload["current"]
                state.temperature = payload["temperature"]
                state.error = payload["error"]
                state.telemetry_ok = True
                state.last_update = payload["last_update"]
                if not state.target_initialized:
                    state.target = clamp(state.position, spec.min_deg, spec.max_deg)
                    state.target_initialized = True
            else:
                state.telemetry_ok = False
        return {key: MotorState(**vars(value)) for key, value in self.states.items()}

    def set_target(self, key: str, value: float) -> float:
        spec = self.specs[key]
        clamped = clamp(value, spec.min_deg, spec.max_deg)
        state = self.states[key]
        state.target = clamped
        state.target_initialized = True
        return clamped

    def command_motor(self, key: str, force: bool = False) -> bool:
        if not self.motion_armed:
            raise RuntimeError("動作鎖定中，先勾選「解除動作鎖定」。")
        spec = self.specs[key]
        state = self.states[key]
        controller = self.buses[spec.bus_key]
        return controller.send_position(
            motor_id=spec.motor_id,
            target_deg=state.target,
            speed_erpm=spec.speed_erpm,
            rpa=spec.rpa,
            force=force,
        )

    def hold_motor(self, key: str) -> None:
        spec = self.specs[key]
        state = self.states[key]
        controller = self.buses[spec.bus_key]
        controller.hold_position(spec.motor_id, fallback=state.target)
        state.target = clamp(state.position, spec.min_deg, spec.max_deg)

    def hold_all(self) -> None:
        fallback_by_bus: Dict[str, Dict[int, float]] = {bus_key: {} for bus_key in BUS_KEYS}
        for spec in self.specs.values():
            fallback_by_bus[spec.bus_key][spec.motor_id] = self.states[spec.key].target
        for bus_key, controller in self.buses.items():
            controller.hold_all(fallbacks=fallback_by_bus[bus_key])

    def set_zero(self, key: str) -> None:
        spec = self.specs[key]
        controller = self.buses[spec.bus_key]
        controller.set_zero(spec.motor_id)
        state = self.states[key]
        state.target = 0.0
        state.target_initialized = True

    def set_zero_all(self) -> None:
        for key in self.specs:
            self.set_zero(key)

    def emergency_stop(self) -> None:
        self.motion_armed = False
        self.hold_all()


class MotorCard(QFrame):
    apply_requested = pyqtSignal(str)
    hold_requested = pyqtSignal(str)
    zero_requested = pyqtSignal(str)
    target_changed = pyqtSignal(str, float)
    spec_changed = pyqtSignal(object)

    def __init__(self, spec: MotorSpec):
        super().__init__()
        self.spec = spec
        self._setting_target = False
        self._target_timer = QTimer(self)
        self._target_timer.setSingleShot(True)
        self._target_timer.setInterval(180)
        self._target_timer.timeout.connect(self._emit_target_changed)
        self._build_ui()
        self.apply_spec(spec)

    def _build_ui(self) -> None:
        self.setObjectName("MotorCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        top_row = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setObjectName("CardTitle")
        self.meta_label = QLabel()
        self.meta_label.setObjectName("MetaLabel")
        self.status_badge = QLabel("OFFLINE")
        self.status_badge.setObjectName("StatusBadge")
        top_row.addWidget(self.title_label)
        top_row.addStretch(1)
        top_row.addWidget(self.meta_label)
        top_row.addWidget(self.status_badge)
        layout.addLayout(top_row)

        telemetry_grid = QGridLayout()
        telemetry_grid.setHorizontalSpacing(14)
        telemetry_grid.setVerticalSpacing(8)
        self.position_value = self._build_metric(telemetry_grid, 0, "Actual")
        self.speed_value = self._build_metric(telemetry_grid, 1, "Speed")
        self.current_value = self._build_metric(telemetry_grid, 2, "Current")
        self.temp_value = self._build_metric(telemetry_grid, 3, "Temp")
        layout.addLayout(telemetry_grid)

        self.error_label = QLabel("Error code: 0")
        self.error_label.setObjectName("ErrorLabel")
        layout.addWidget(self.error_label)

        range_row = QGridLayout()
        range_row.setHorizontalSpacing(12)
        self.bus_combo = self._bus_combo()
        self.id_spin = self._id_spinbox()
        self.min_spin = self._limit_spinbox()
        self.max_spin = self._limit_spinbox()
        self.speed_spin = self._command_spinbox(1_000, 50_000, 500)
        self.rpa_spin = self._command_spinbox(1_000, 50_000, 1_000)
        if len(BUS_KEYS) > 1:
            range_row.addWidget(QLabel("Bus"), 0, 0)
            range_row.addWidget(self.bus_combo, 0, 1)
            range_row.addWidget(QLabel("ID"), 0, 2)
            range_row.addWidget(self.id_spin, 0, 3)
            range_row.addWidget(QLabel("Min"), 1, 0)
            range_row.addWidget(self.min_spin, 1, 1)
            range_row.addWidget(QLabel("Max"), 1, 2)
            range_row.addWidget(self.max_spin, 1, 3)
            range_row.addWidget(QLabel("Speed"), 2, 0)
            range_row.addWidget(self.speed_spin, 2, 1)
            range_row.addWidget(QLabel("RPA"), 2, 2)
            range_row.addWidget(self.rpa_spin, 2, 3)
        else:
            range_row.addWidget(QLabel("ID"), 0, 0)
            range_row.addWidget(self.id_spin, 0, 1)
            range_row.addWidget(QLabel("Min"), 0, 2)
            range_row.addWidget(self.min_spin, 0, 3)
            range_row.addWidget(QLabel("Max"), 1, 0)
            range_row.addWidget(self.max_spin, 1, 1)
            range_row.addWidget(QLabel("Speed"), 1, 2)
            range_row.addWidget(self.speed_spin, 1, 3)
            range_row.addWidget(QLabel("RPA"), 2, 0)
            range_row.addWidget(self.rpa_spin, 2, 1)
        layout.addLayout(range_row)

        target_row = QHBoxLayout()
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setDecimals(1)
        self.target_spin.setSingleStep(0.5)
        self.target_spin.valueChanged.connect(self._on_target_spin_changed)
        self.target_spin.editingFinished.connect(self._emit_target_changed)
        self.target_slider = QSlider(Qt.Orientation.Horizontal)
        self.target_slider.setTracking(False)
        self.target_slider.sliderReleased.connect(self._emit_target_changed)
        self.target_slider.valueChanged.connect(self._on_slider_changed)
        target_row.addWidget(QLabel("Target"))
        target_row.addWidget(self.target_slider, 1)
        target_row.addWidget(self.target_spin)
        layout.addLayout(target_row)

        self.temp_bar = QProgressBar()
        self.temp_bar.setRange(0, 100)
        self.temp_bar.setFormat("Temperature %v°C")
        self.temp_bar.setValue(0)
        layout.addWidget(self.temp_bar)

        quick_row = QHBoxLayout()
        for label, delta in (("-5°", -5.0), ("-1°", -1.0), ("+1°", 1.0), ("+5°", 5.0)):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, d=delta: self.nudge_target(d))
            quick_row.addWidget(button)
        layout.addLayout(quick_row)

        action_row = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.hold_button = QPushButton("Hold")
        self.zero_button = QPushButton("Set Zero")
        self.apply_button.clicked.connect(lambda: self.apply_requested.emit(self.spec.key))
        self.hold_button.clicked.connect(lambda: self.hold_requested.emit(self.spec.key))
        self.zero_button.clicked.connect(lambda: self.zero_requested.emit(self.spec.key))
        action_row.addWidget(self.apply_button)
        action_row.addWidget(self.hold_button)
        action_row.addWidget(self.zero_button)
        layout.addLayout(action_row)

    def _build_metric(self, grid: QGridLayout, column: int, caption: str) -> QLabel:
        title = QLabel(caption)
        title.setObjectName("MetricCaption")
        value = QLabel("--")
        value.setObjectName("MetricValue")
        grid.addWidget(title, 0, column)
        grid.addWidget(value, 1, column)
        return value

    def _bus_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.addItems(BUS_KEYS)
        combo.currentTextChanged.connect(self._emit_spec_changed)
        return combo

    def _id_spinbox(self) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 255)
        spin.valueChanged.connect(self._emit_spec_changed)
        return spin

    def _limit_spinbox(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-36_000.0, 36_000.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.valueChanged.connect(self._on_limit_changed)
        return spin

    def _command_spinbox(self, minimum: int, maximum: int, step: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.valueChanged.connect(self._emit_spec_changed)
        return spin

    def apply_spec(self, spec: MotorSpec) -> None:
        self.spec = spec
        self.title_label.setText(f"{spec.name}  {spec.family}")
        self.meta_label.setText(f"Motor ID {spec.motor_id}")

        for widget, value in (
            (self.bus_combo, spec.bus_key),
            (self.id_spin, spec.motor_id),
            (self.min_spin, spec.min_deg),
            (self.max_spin, spec.max_deg),
            (self.speed_spin, spec.speed_erpm),
            (self.rpa_spin, spec.rpa),
        ):
            blocker = QSignalBlocker(widget)
            if isinstance(widget, QComboBox):
                widget.setCurrentText(value)
            else:
                widget.setValue(value)
            del blocker

        self._apply_target_range(spec.min_deg, spec.max_deg)
        self.setStyleSheet(
            f"QFrame#MotorCard {{ border: 1px solid rgba(255,255,255,0.08); "
            f"border-left: 5px solid {spec.accent}; border-radius: 22px; "
            f"background: rgba(14, 22, 37, 0.92); }}"
        )

    def current_spec(self) -> MotorSpec:
        lower = min(self.min_spin.value(), self.max_spin.value() - 0.5)
        upper = max(self.max_spin.value(), lower + 0.5)
        return MotorSpec(
            key=self.spec.key,
            name=self.spec.name,
            family=self.spec.family,
            bus_key=self.bus_combo.currentText() if len(BUS_KEYS) > 1 else BUS_KEYS[0],
            motor_id=self.id_spin.value(),
            min_deg=lower,
            max_deg=upper,
            speed_erpm=self.speed_spin.value(),
            rpa=self.rpa_spin.value(),
            accent=self.spec.accent,
        )

    def set_state(self, spec: MotorSpec, state: MotorState) -> None:
        self.spec = spec
        self.title_label.setText(f"{spec.name}  {spec.family}")
        self.meta_label.setText(f"Motor ID {spec.motor_id}")
        self.position_value.setText(f"{state.position:,.1f}°")
        self.speed_value.setText(f"{state.speed:,.0f} eRPM")
        self.current_value.setText(f"{state.current:,.2f} A")
        self.temp_value.setText(f"{state.temperature:,.0f}°C")
        self.error_label.setText(f"Error code: {state.error}")
        self.temp_bar.setValue(int(clamp(state.temperature, 0, 100)))
        self.temp_bar.setStyleSheet(
            "QProgressBar::chunk { background-color: %s; border-radius: 6px; }"
            % ("#ff5f56" if state.temperature >= 70 else spec.accent)
        )

        if state.target_initialized and not self._setting_target:
            self.set_target_value(state.target)

        if not state.bus_connected:
            badge_text = "OFFLINE"
            badge_color = "#7d8ba7"
        elif state.error:
            badge_text = f"FAULT {state.error}"
            badge_color = "#ff5f56"
        elif not state.telemetry_ok:
            badge_text = "WAITING"
            badge_color = "#ffbf47"
        elif state.temperature >= 70:
            badge_text = "HOT"
            badge_color = "#ff8f3d"
        else:
            badge_text = "READY"
            badge_color = spec.accent

        self.status_badge.setText(badge_text)
        self.status_badge.setStyleSheet(
            f"background:{badge_color}; color:#08111f; padding:6px 10px; "
            "border-radius:12px; font-weight:700;"
        )

    def set_target_value(self, value: float) -> None:
        self._setting_target = True
        clamped = clamp(value, self.min_spin.value(), self.max_spin.value())
        for widget, widget_value in (
            (self.target_spin, clamped),
            (self.target_slider, int(round(clamped * SLIDER_SCALE))),
        ):
            blocker = QSignalBlocker(widget)
            widget.setValue(widget_value)
            del blocker
        self._setting_target = False

    def nudge_target(self, delta: float) -> None:
        self.set_target_value(self.target_spin.value() + delta)
        self._target_timer.start()

    def _apply_target_range(self, lower: float, upper: float) -> None:
        slider_blocker = QSignalBlocker(self.target_slider)
        spin_blocker = QSignalBlocker(self.target_spin)
        self.target_slider.setRange(
            int(round(lower * SLIDER_SCALE)),
            int(round(upper * SLIDER_SCALE)),
        )
        self.target_spin.setRange(lower, upper)
        del slider_blocker
        del spin_blocker

    def _on_target_spin_changed(self, value: float) -> None:
        if self._setting_target:
            return
        self._setting_target = True
        blocker = QSignalBlocker(self.target_slider)
        self.target_slider.setValue(int(round(value * SLIDER_SCALE)))
        del blocker
        self._setting_target = False
        self._target_timer.start()

    def _on_slider_changed(self, raw_value: int) -> None:
        if self._setting_target:
            return
        self._setting_target = True
        blocker = QSignalBlocker(self.target_spin)
        self.target_spin.setValue(raw_value / SLIDER_SCALE)
        del blocker
        self._setting_target = False

    def _on_limit_changed(self) -> None:
        if self.min_spin.value() >= self.max_spin.value():
            sender = self.sender()
            if sender is self.min_spin:
                self.max_spin.setValue(self.min_spin.value() + 0.5)
            else:
                self.min_spin.setValue(self.max_spin.value() - 0.5)
        self._apply_target_range(self.min_spin.value(), self.max_spin.value())
        self.set_target_value(self.target_spin.value())
        self._emit_spec_changed()

    def _emit_target_changed(self) -> None:
        self.target_changed.emit(self.spec.key, self.target_spin.value())

    def _emit_spec_changed(self) -> None:
        self.meta_label.setText(f"Motor ID {self.id_spin.value()}")
        self.spec_changed.emit(self.current_spec())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = ArmController(DEFAULT_MOTORS)
        self.cards: Dict[str, MotorCard] = {}
        self._emergency_latched = False
        self._latest_states: Dict[str, MotorState] = {}
        self.setWindowTitle(APP_TITLE)
        self.resize(1480, 920)
        self._build_ui()
        self._apply_style()
        self.refresh_ports()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(STATUS_POLL_MS)
        self._poll_timer.timeout.connect(self._refresh_ui)
        self._poll_timer.start()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(18)

        header = QFrame()
        header.setObjectName("HeaderCard")
        header_layout = QVBoxLayout(header)
        title = QLabel("Robotic Arm Motion Desk")
        title.setObjectName("HeaderTitle")
        subtitle = QLabel("三軸手動控制介面，內建軟體限位、Arm 鎖、Hold、過熱保護與急停。")
        subtitle.setObjectName("HeaderSubtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root_layout.addWidget(header)

        control_row = QHBoxLayout()
        control_row.setSpacing(18)
        control_row.addWidget(self._build_system_panel(), 3)
        control_row.addWidget(self._build_log_panel(), 1)
        root_layout.addLayout(control_row)

        motor_area = QScrollArea()
        motor_area.setWidgetResizable(True)
        motor_host = QWidget()
        motor_layout = QHBoxLayout(motor_host)
        motor_layout.setContentsMargins(0, 0, 0, 0)
        motor_layout.setSpacing(18)

        for spec in DEFAULT_MOTORS:
            card = MotorCard(spec)
            card.apply_requested.connect(self._apply_single)
            card.hold_requested.connect(self._hold_single)
            card.zero_requested.connect(self._zero_single)
            card.target_changed.connect(self._target_changed)
            card.spec_changed.connect(self._spec_changed)
            self.cards[spec.key] = card
            motor_layout.addWidget(card, 1)

        motor_area.setWidget(motor_host)
        root_layout.addWidget(motor_area, 1)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("尚未連線")

    def _build_system_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SideCard")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("System")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.port_boxes: Dict[str, QComboBox] = {}
        for bus_key in BUS_KEYS:
            row = QHBoxLayout()
            row.addWidget(QLabel("USB / COM" if len(BUS_KEYS) == 1 else f"Bus {bus_key}"))
            combo = QComboBox()
            combo.setEditable(True)
            combo.setMinimumWidth(140)
            combo.currentTextChanged.connect(lambda _text, b=bus_key: self._update_status_hint(b))
            self.port_boxes[bus_key] = combo
            row.addWidget(combo, 1)
            layout.addLayout(row)

        refresh_button = QPushButton("Refresh COM")
        refresh_button.clicked.connect(self.refresh_ports)
        layout.addWidget(refresh_button)

        self.arm_checkbox = QCheckBox("解除動作鎖定")
        self.arm_checkbox.toggled.connect(self._arm_toggled)
        layout.addWidget(self.arm_checkbox)

        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temp Limit"))
        self.temp_limit_spin = QSpinBox()
        self.temp_limit_spin.setRange(40, 120)
        self.temp_limit_spin.setValue(DEFAULT_TEMP_LIMIT_C)
        temp_row.addWidget(self.temp_limit_spin)
        layout.addLayout(temp_row)

        button_row = QGridLayout()
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        self.apply_all_button = QPushButton("Apply All")
        self.hold_all_button = QPushButton("Hold All")
        self.zero_all_button = QPushButton("Set Zero All")
        self.estop_button = QPushButton("E-STOP")
        self.estop_button.setObjectName("DangerButton")
        self.connect_button.clicked.connect(self._connect_all)
        self.disconnect_button.clicked.connect(self._disconnect_all)
        self.apply_all_button.clicked.connect(self._apply_all)
        self.hold_all_button.clicked.connect(self._hold_all)
        self.zero_all_button.clicked.connect(self._zero_all)
        self.estop_button.clicked.connect(lambda: self._trigger_emergency_stop("使用者觸發急停"))
        button_row.addWidget(self.connect_button, 0, 0)
        button_row.addWidget(self.disconnect_button, 0, 1)
        button_row.addWidget(self.apply_all_button, 1, 0)
        button_row.addWidget(self.hold_all_button, 1, 1)
        button_row.addWidget(self.zero_all_button, 2, 0)
        button_row.addWidget(self.estop_button, 2, 1)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return panel

    def _build_log_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SideCard")
        panel.setMaximumWidth(360)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title = QLabel("Log")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        note = QLabel("只保留最近訊息。先連線確認回授，再解鎖 Arm。")
        note.setWordWrap(True)
        note.setObjectName("HintLabel")
        layout.addWidget(note)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(110)
        self.log_view.setMaximumHeight(150)
        self.log_view.document().setMaximumBlockCount(60)
        layout.addWidget(self.log_view, 1)
        return panel

    def _apply_style(self) -> None:
        self.setFont(QFont("Bahnschrift SemiCondensed", 10))
        self.setStyleSheet(
            """
            QWidget { color: #e9eef8; background: #08111f; }
            QMainWindow {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #08111f, stop:0.45 #0c1728, stop:1 #10233c
                );
            }
            QFrame#HeaderCard, QFrame#SideCard {
                background: rgba(9, 19, 33, 0.86);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 24px;
            }
            QLabel#HeaderTitle { font-size: 30px; font-weight: 700; color: #f8fbff; }
            QLabel#HeaderSubtitle, QLabel#HintLabel { color: #9eb0ca; font-size: 13px; }
            QLabel#SectionTitle, QLabel#CardTitle { font-size: 18px; font-weight: 700; color: #f8fbff; }
            QLabel#MetaLabel { color: #8ea2c0; }
            QLabel#MetricCaption { color: #89a0be; font-size: 11px; }
            QLabel#MetricValue { font-size: 18px; font-weight: 700; color: #f8fbff; }
            QLabel#ErrorLabel { color: #ffbf47; font-size: 12px; }
            QPushButton {
                background: #153252;
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 14px;
                padding: 10px 14px;
                font-weight: 700;
            }
            QPushButton:hover { background: #1d456e; }
            QPushButton:pressed { background: #112a45; }
            QPushButton#DangerButton { background: #ba3c30; }
            QPushButton#DangerButton:hover { background: #d94d41; }
            QCheckBox { spacing: 10px; font-weight: 700; }
            QCheckBox::indicator {
                width: 18px; height: 18px; border-radius: 5px;
                border: 1px solid rgba(255,255,255,0.16); background: #0f2035;
            }
            QCheckBox::indicator:checked { background: #35c2a1; }
            QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit {
                background: #0f2035;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 12px;
                padding: 8px 10px;
            }
            QSlider::groove:horizontal {
                background: rgba(255,255,255,0.10);
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #f8fbff;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QProgressBar {
                background: #0f2035;
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 8px;
                padding: 2px;
                text-align: center;
            }
            QStatusBar { background: rgba(9,19,33,0.92); color: #9eb0ca; }
            """
        )

    def refresh_ports(self) -> None:
        available_ports = [port.device for port in list_ports.comports()]
        for bus_key, combo in self.port_boxes.items():
            previous = combo.currentText() or DEFAULT_PORTS.get(bus_key, "")
            blocker = QSignalBlocker(combo)
            combo.clear()
            combo.addItem("")
            combo.addItems(available_ports)
            combo.setCurrentText(previous)
            del blocker
        self.statusBar().showMessage(
            f"COM ports: {', '.join(available_ports) if available_ports else '沒有偵測到裝置'}",
            4000,
        )

    def _spec_changed(self, spec: MotorSpec) -> None:
        self.controller.update_spec(spec)

    def _target_changed(self, key: str, value: float) -> None:
        clamped = self.controller.set_target(key, value)
        self.cards[key].set_target_value(clamped)

    def _apply_single(self, key: str) -> None:
        self._send_command(key, force=True)

    def _apply_all(self) -> None:
        for key in self.cards:
            self._send_command(key, force=True)

    def _hold_single(self, key: str) -> None:
        try:
            self.controller.hold_motor(key)
            self._log(f"{self.controller.specs[key].name}: Hold 目前位置")
        except Exception as exc:
            self._log(str(exc))

    def _hold_all(self) -> None:
        try:
            self.controller.hold_all()
            self._log("所有馬達進入 Hold")
        except Exception as exc:
            self._log(str(exc))

    def _zero_single(self, key: str) -> None:
        spec = self.controller.specs[key]
        answer = QMessageBox.question(
            self,
            "確認歸零",
            f"確定要將 {spec.name} (ID {spec.motor_id}) 設為零點嗎？\n請先確認機械手臂姿態安全。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.controller.set_zero(key)
            self.cards[key].set_target_value(0.0)
            self._log(f"{spec.name}: 已送出 Set Zero")
        except Exception as exc:
            self._log(str(exc))

    def _zero_all(self) -> None:
        answer = QMessageBox.question(
            self,
            "確認全部歸零",
            "確定要將三顆馬達都設為零點嗎？\n請先確認機構固定且沒有負載風險。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.controller.set_zero_all()
            for card in self.cards.values():
                card.set_target_value(0.0)
            self._log("已送出全部 Set Zero")
        except Exception as exc:
            self._log(str(exc))

    def _send_command(self, key: str, force: bool) -> None:
        try:
            sent = self.controller.command_motor(key, force=force)
            if sent:
                spec = self.controller.specs[key]
                target = self.controller.states[key].target
                self._log(f"{spec.name}: 目標 {target:.1f}°")
        except Exception as exc:
            self._log(str(exc))

    def _connect_all(self) -> None:
        self._emergency_latched = False
        try:
            self._apply_card_specs()
            port_map = {
                bus_key: combo.currentText().strip()
                for bus_key, combo in self.port_boxes.items()
            }
            self.controller.connect_buses(port_map)
            connected_ports = [
                port_map[bus_key]
                for bus_key, controller in self.controller.buses.items()
                if controller.connected
            ]
            if len(BUS_KEYS) == 1 and connected_ports:
                self._log(f"已連線 USB: {connected_ports[0]}")
            else:
                self._log(
                    "已連線: "
                    + ", ".join(
                        f"Bus {bus_key} -> {port_map[bus_key]}"
                        for bus_key, controller in self.controller.buses.items()
                        if controller.connected
                    )
                )
        except Exception as exc:
            self._log(f"連線失敗: {exc}")
            self.controller.disconnect_all()

    def _disconnect_all(self) -> None:
        self.arm_checkbox.setChecked(False)
        self.controller.disconnect_all()
        self._log("已斷線")

    def _arm_toggled(self, checked: bool) -> None:
        if checked:
            reason = self._current_safety_blocker()
            if reason:
                blocker = QSignalBlocker(self.arm_checkbox)
                self.arm_checkbox.setChecked(False)
                del blocker
                self.controller.set_motion_armed(False)
                self._log(f"無法 Arm: {reason}")
                return
        self.controller.set_motion_armed(checked)
        self._log("動作已解鎖，可送出 Apply" if checked else "動作已鎖定")

    def _refresh_ui(self) -> None:
        try:
            self._apply_card_specs(update_targets=False)
            states = self.controller.refresh_states()
        except Exception as exc:
            self._log(f"狀態更新失敗: {exc}")
            return

        self._latest_states = states
        for key, card in self.cards.items():
            spec = self.controller.specs[key]
            card.set_state(spec, states[key])
            self._check_safety(spec, states[key])

        arm_state = "ARMED" if self.controller.motion_armed else "SAFE"
        self.statusBar().showMessage(
            f"{arm_state} | Poll {STATUS_POLL_MS} ms | Temp Limit {self.temp_limit_spin.value()}°C"
        )

    def _check_safety(self, spec: MotorSpec, state: MotorState) -> None:
        if not state.bus_connected or not state.telemetry_ok:
            return
        if state.error and (self.arm_checkbox.isChecked() or not self._emergency_latched):
            self._trigger_emergency_stop(f"{spec.name} 回報錯誤碼 {state.error}")
            return
        if state.temperature >= self.temp_limit_spin.value() and (
            self.arm_checkbox.isChecked() or not self._emergency_latched
        ):
            self._trigger_emergency_stop(
                f"{spec.name} 溫度 {state.temperature:.0f}°C 超過限制 {self.temp_limit_spin.value()}°C"
            )

    def _trigger_emergency_stop(self, reason: str) -> None:
        self._emergency_latched = True
        self.arm_checkbox.setChecked(False)
        try:
            self.controller.emergency_stop()
        except Exception:
            pass
        self._log(f"[E-STOP] {reason}")
        self.statusBar().showMessage(f"E-STOP | {reason}")

    def _apply_card_specs(self, update_targets: bool = True) -> None:
        for key, card in self.cards.items():
            spec = card.current_spec()
            self.controller.update_spec(spec)
            if update_targets:
                clamped = self.controller.set_target(key, card.target_spin.value())
                card.set_target_value(clamped)

    def _current_safety_blocker(self) -> str | None:
        any_connected = any(controller.connected for controller in self.controller.buses.values())
        if not any_connected:
            return "尚未連線到任何 CAN bus"
        if not self._latest_states:
            return "等待回授初始化"
        for key, state in self._latest_states.items():
            spec = self.controller.specs[key]
            if state.bus_connected and not state.telemetry_ok:
                return f"{spec.name} 尚未收到回授"
            if not state.bus_connected or not state.telemetry_ok:
                continue
            if state.error:
                return f"{spec.name} 錯誤碼 {state.error}"
            if state.temperature >= self.temp_limit_spin.value():
                return f"{spec.name} 溫度 {state.temperature:.0f}°C"
        return None

    def _update_status_hint(self, bus_key: str) -> None:
        text = self.port_boxes[bus_key].currentText().strip()
        if text:
            if len(BUS_KEYS) == 1:
                self.statusBar().showMessage(f"USB 準備使用 {text}")
            else:
                self.statusBar().showMessage(f"Bus {bus_key} 準備使用 {text}")

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")
        self.statusBar().showMessage(message, 5000)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self.controller.hold_all()
        except Exception:
            pass
        self.controller.disconnect_all()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
