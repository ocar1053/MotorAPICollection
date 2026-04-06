from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import serial

import cubemotorAK109Util as motor_proto

CAN_BAUDRATE = 2_000_000
SERIAL_TIMEOUT = 0.05
WRITE_TIMEOUT = 0.20
STATUS_POLL_MS = 150
COMMAND_RATE_LIMIT_S = 0.08
SAFE_HOLD_SPEED_ERPM = 3_000
SAFE_HOLD_RPA = 10_000
HOLD_KEEPALIVE_MS = 100
AUTO_HOLD_POSITION_TOLERANCE_DEG = 0.5
AUTO_HOLD_SPEED_TOLERANCE_ERPM = 300.0
ZERO_VERIFY_TOLERANCE_DEG = 2.0
ZERO_VERIFY_TIMEOUT_S = 1.5
ZERO_COMMAND_SETTLE_S = 0.15
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
    gear_ratio: float
    accent: str


@dataclass
class MotorState:
    target: float = 0.0
    commanded_target: float = 0.0
    position: float = 0.0
    motor_position: float = 0.0
    speed: float = 0.0
    current: float = 0.0
    temperature: float = 0.0
    error: int = 0
    bus_connected: bool = False
    telemetry_ok: bool = False
    target_initialized: bool = False
    last_update: float = 0.0
    auto_hold_pending: bool = False
    auto_hold_engaged: bool = False


DEFAULT_MOTORS = [
    MotorSpec("joint_a", "Joint A", "AK10-9", "A", 0, -75.0, 0.0, 6_000, 12_000, 2.0, "#35c2a1"),
    MotorSpec("joint_b", "Joint B", "AK10-9", "A", 1, -90.0, 90.0, 6_000, 12_000, 2.0, "#ff8f3d"),
    MotorSpec("joint_c", "Joint C", "AK70", "A", 93, -90.0, 90.0, 6_000, 12_000, 2.0, "#5bbcff"),
]

DEFAULT_PORTS = {"A": "COM12"}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def joint_to_motor_deg(spec: MotorSpec, joint_deg: float) -> float:
    return joint_deg * spec.gear_ratio


def motor_to_joint_deg(spec: MotorSpec, motor_deg: float) -> float:
    if spec.gear_ratio == 0:
        return motor_deg
    return motor_deg / spec.gear_ratio


def build_bridge_init_frame() -> bytearray:
    frame = [
        0xAA,
        0x55,
        0x12,
        0x01,
        0x02,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
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
        self._last_command_ts: dict[int, float] = {}
        self._states: dict[int, dict] = {}

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

    def wait_for_position_close(
        self,
        motor_id: int,
        target_deg: float,
        tolerance_deg: float,
        timeout_s: float,
    ) -> dict | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            payload = self.snapshot_states().get(motor_id)
            if payload and abs(payload["position"] - target_deg) <= tolerance_deg:
                return payload
            time.sleep(0.05)
        return None

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
        self.send_hold_target(motor_id, target_deg=current_position, force=True)

    def send_hold_target(self, motor_id: int, target_deg: float, force: bool = False) -> bool:
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
                speed_erpm=SAFE_HOLD_SPEED_ERPM,
                rpa=SAFE_HOLD_RPA,
            )
            self._last_command_ts[motor_id] = now
            return True

    def hold_all(self, fallbacks: dict[int, float] | None = None) -> None:
        fallbacks = fallbacks or {}
        for motor_id in list(self.motor_ids):
            try:
                self.hold_position(motor_id, fallback=fallbacks.get(motor_id, 0.0))
            except Exception:
                continue

    def snapshot_states(self) -> dict[int, dict]:
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
            raise RuntimeError(f"Bus {self.bus_key} is not connected.")


class ArmController:
    def __init__(self, specs: list[MotorSpec]):
        self.specs: dict[str, MotorSpec] = {spec.key: spec for spec in specs}
        self.states: dict[str, MotorState] = {
            spec.key: MotorState(target=0.0) for spec in specs
        }
        self.buses = {bus_key: CanBusController(bus_key) for bus_key in BUS_KEYS}
        self.motion_armed = False

    def update_spec(self, spec: MotorSpec) -> None:
        self.specs[spec.key] = spec
        state = self.states[spec.key]
        state.target = clamp(state.target, spec.min_deg, spec.max_deg)

    def _clear_auto_hold_state(self, key: str) -> None:
        state = self.states[key]
        state.auto_hold_pending = False
        state.auto_hold_engaged = False

    def _arm_auto_hold(self, key: str) -> None:
        state = self.states[key]
        state.auto_hold_pending = True
        state.auto_hold_engaged = False

    def set_motion_armed(self, armed: bool) -> None:
        self.motion_armed = armed
        if not armed:
            for key in self.states:
                self._clear_auto_hold_state(key)

    def connect_buses(self, port_map: dict[str, str]) -> None:
        assignments: dict[str, list[int]] = {bus_key: [] for bus_key in BUS_KEYS}
        for spec in self.specs.values():
            assignments[spec.bus_key].append(spec.motor_id)

        for bus_key, motor_ids in assignments.items():
            if len(motor_ids) != len(set(motor_ids)):
                raise RuntimeError(f"Bus {bus_key} has duplicate motor IDs.")

        for bus_key, motor_ids in assignments.items():
            controller = self.buses[bus_key]
            if not motor_ids:
                controller.disconnect()
                continue
            port = port_map.get(bus_key, "").strip()
            if not port:
                raise RuntimeError(f"Bus {bus_key} is missing a COM port.")
            controller.connect(port, motor_ids)

        for state in self.states.values():
            state.target_initialized = False
            state.telemetry_ok = False
            state.auto_hold_pending = False
            state.auto_hold_engaged = False

    def disconnect_all(self) -> None:
        for controller in self.buses.values():
            controller.disconnect()
        for state in self.states.values():
            state.bus_connected = False
            state.telemetry_ok = False
            state.auto_hold_pending = False
            state.auto_hold_engaged = False

    def refresh_states(self) -> dict[str, MotorState]:
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
                state.motor_position = payload["position"]
                state.position = motor_to_joint_deg(spec, payload["position"])
                state.speed = payload["speed"]
                state.current = payload["current"]
                state.temperature = payload["temperature"]
                state.error = payload["error"]
                state.telemetry_ok = True
                state.last_update = payload["last_update"]
                if not state.target_initialized:
                    current_position = clamp(state.position, spec.min_deg, spec.max_deg)
                    state.target = current_position
                    state.commanded_target = current_position
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
            raise RuntimeError("Motion is not armed.")
        spec = self.specs[key]
        state = self.states[key]
        controller = self.buses[spec.bus_key]
        commanded_target = state.target
        motor_target = joint_to_motor_deg(spec, commanded_target)
        sent = controller.send_position(
            motor_id=spec.motor_id,
            target_deg=motor_target,
            speed_erpm=spec.speed_erpm,
            rpa=spec.rpa,
            force=force,
        )
        if sent:
            state.commanded_target = commanded_target
            self._arm_auto_hold(key)
        return sent

    def command_home_zero(self, key: str) -> bool:
        if not self.motion_armed:
            raise RuntimeError("Motion is not armed.")
        spec = self.specs[key]
        controller = self.buses[spec.bus_key]
        target = self.set_target(key, 0.0)
        motor_target = joint_to_motor_deg(spec, target)
        sent = controller.send_position(
            motor_id=spec.motor_id,
            target_deg=motor_target,
            speed_erpm=spec.speed_erpm,
            rpa=spec.rpa,
            force=True,
        )
        if sent:
            state = self.states[key]
            state.commanded_target = target
            self._arm_auto_hold(key)
        return sent

    def hold_motor(self, key: str) -> None:
        spec = self.specs[key]
        state = self.states[key]
        controller = self.buses[spec.bus_key]
        state.target = clamp(state.position, spec.min_deg, spec.max_deg)
        state.commanded_target = state.target
        state.target_initialized = True
        controller.send_hold_target(
            spec.motor_id,
            target_deg=joint_to_motor_deg(spec, state.commanded_target),
            force=True,
        )
        state.auto_hold_pending = False
        state.auto_hold_engaged = True

    def hold_all(self) -> None:
        fallback_by_bus: dict[str, dict[int, float]] = {bus_key: {} for bus_key in BUS_KEYS}
        for spec in self.specs.values():
            fallback_by_bus[spec.bus_key][spec.motor_id] = joint_to_motor_deg(
                spec,
                self.states[spec.key].commanded_target,
            )
        for bus_key, controller in self.buses.items():
            controller.hold_all(fallbacks=fallback_by_bus[bus_key])

    def set_zero(self, key: str) -> None:
        spec = self.specs[key]
        controller = self.buses[spec.bus_key]
        controller.set_zero(spec.motor_id)
        time.sleep(ZERO_COMMAND_SETTLE_S)
        payload = controller.wait_for_position_close(
            spec.motor_id,
            target_deg=0.0,
            tolerance_deg=ZERO_VERIFY_TOLERANCE_DEG * spec.gear_ratio,
            timeout_s=ZERO_VERIFY_TIMEOUT_S,
        )
        if payload is None:
            raise RuntimeError("Zero command was sent, but telemetry did not settle near zero.")
        state = self.states[key]
        state.motor_position = float(payload["position"])
        state.position = motor_to_joint_deg(spec, state.motor_position)
        state.target = clamp(0.0, spec.min_deg, spec.max_deg)
        state.commanded_target = state.target
        state.target_initialized = True
        self._clear_auto_hold_state(key)

    def set_zero_all(self) -> None:
        for key in self.specs:
            self.set_zero(key)

    def emergency_stop(self) -> None:
        self.motion_armed = False
        for key in self.states:
            self._clear_auto_hold_state(key)
        self.hold_all()

    def try_auto_hold(
        self,
        key: str,
        position_tolerance_deg: float = AUTO_HOLD_POSITION_TOLERANCE_DEG,
        speed_tolerance_erpm: float = AUTO_HOLD_SPEED_TOLERANCE_ERPM,
    ) -> bool:
        state = self.states[key]
        if not state.auto_hold_pending or not self.motion_armed:
            return False
        if not state.bus_connected or not state.telemetry_ok or state.error:
            return False
        if abs(state.commanded_target - state.position) > position_tolerance_deg:
            return False
        if abs(state.speed) > speed_tolerance_erpm:
            return False
        spec = self.specs[key]
        controller = self.buses[spec.bus_key]
        sent = controller.send_hold_target(
            spec.motor_id,
            target_deg=joint_to_motor_deg(spec, state.commanded_target),
            force=True,
        )
        if sent:
            state.auto_hold_pending = False
            state.auto_hold_engaged = True
        return sent

    def maintain_target_holds(self) -> None:
        if not self.motion_armed:
            return
        for key, state in self.states.items():
            spec = self.specs[key]
            controller = self.buses[spec.bus_key]
            if state.error or not controller.connected:
                continue
            if state.auto_hold_pending:
                try:
                    controller.send_position(
                        motor_id=spec.motor_id,
                        target_deg=joint_to_motor_deg(spec, state.commanded_target),
                        speed_erpm=spec.speed_erpm,
                        rpa=spec.rpa,
                    )
                except Exception:
                    continue
                continue
            if not state.auto_hold_engaged:
                continue
            try:
                controller.send_hold_target(
                    spec.motor_id,
                    target_deg=joint_to_motor_deg(spec, state.commanded_target),
                    force=True,
                )
            except Exception:
                continue
