import argparse
import time

import serial

import cubemotorAK109Util as ak109_util
import cubemotorAK70Util as ak70_util


CAN_BAUDRATE = 2_000_000
SERIAL_TIMEOUT = 0.05
WRITE_TIMEOUT = 0.20
MOVE_COMMAND_PERIOD_S = 0.10
DEFAULT_HOLD_COMMAND_PERIOD_S = 0.05
STATUS_PRINT_PERIOD_S = 0.50


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


class MotorAdapter:
    def __init__(self, family: str, ser: serial.Serial, motor_id: int):
        self.family = family
        self.ser = ser
        self.motor_id = motor_id
        if family == "ak70":
            self.util = ak70_util
            self.listener = ak70_util.SerialCanListener(ser)
        else:
            self.util = ak109_util
            self.listener = ak109_util.SerialCanListener(ser)
            self.listener._th.start()

    def close(self) -> None:
        self.listener.close()

    def get_status(self) -> dict | None:
        if self.family == "ak70":
            pos, spd, cur, temp, err, can_id = self.listener.get_status()
            if can_id != self.motor_id:
                return None
            return {
                "position": pos,
                "speed": spd,
                "current": cur,
                "temperature": temp,
                "error": err,
                "can_id": can_id,
            }

        payload = self.listener.get_status().get(self.motor_id)
        if not payload:
            return None
        return {
            "position": float(payload["position"]),
            "speed": float(payload["speed"]),
            "current": float(payload["current"]),
            "temperature": float(payload["temperature"]),
            "error": int(payload["error"]),
            "can_id": self.motor_id,
        }

    def set_zero(self) -> None:
        self.util.servo_mod_set_zero(self.ser, control_mode_id=5, motor_id=self.motor_id)

    def send_position_speed(self, target_deg: float, speed_erpm: int, rpa: int) -> None:
        self.util.servo_mod_pos_speed(
            self.ser,
            control_mode_id=6,
            motor_id=self.motor_id,
            pos_deg=target_deg,
            speed_erpm=speed_erpm,
            rpa=rpa,
        )

    def send_position_only(self, target_deg: float) -> None:
        self.util.servo_mod_pos(
            self.ser,
            control_mode_id=4,
            motor_id=self.motor_id,
            pos_deg=target_deg,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone servo position-hold test for Cubemars AK motors.")
    parser.add_argument("--family", choices=("ak10-9", "ak70"), default="ak10-9")
    parser.add_argument("--port", default="COM12")
    parser.add_argument("--motor-id", type=int, required=True)
    parser.add_argument("--target-deg", type=float, default=10.0)
    parser.add_argument("--speed-erpm", type=int, default=2000)
    parser.add_argument("--rpa", type=int, default=5000)
    parser.add_argument("--position-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--speed-tolerance-erpm", type=float, default=300.0)
    parser.add_argument("--move-timeout-s", type=float, default=8.0)
    parser.add_argument("--hold-seconds", type=float, default=10.0)
    parser.add_argument("--hold-mode", choices=("pos", "pos_speed"), default="pos")
    parser.add_argument("--hold-command-period-s", type=float, default=DEFAULT_HOLD_COMMAND_PERIOD_S)
    parser.add_argument("--hold-speed-erpm", type=int, default=3000)
    parser.add_argument("--hold-rpa", type=int, default=10000)
    parser.add_argument("--set-zero", action="store_true")
    return parser.parse_args()


def wait_for_status(adapter: MotorAdapter, timeout_s: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = adapter.get_status()
        if status is not None:
            return status
        time.sleep(0.02)
    raise RuntimeError("No telemetry received from the selected motor ID.")


def try_wait_for_status(adapter: MotorAdapter, timeout_s: float = 2.0) -> dict | None:
    try:
        return wait_for_status(adapter, timeout_s=timeout_s)
    except RuntimeError:
        return None


def print_status(prefix: str, status: dict, target_deg: float | None = None) -> None:
    target_text = "" if target_deg is None else f" target={target_deg:.1f}deg"
    print(
        f"{prefix}{target_text} "
        f"pos={status['position']:.1f}deg "
        f"spd={status['speed']:.0f}erpm "
        f"cur={status['current']:.2f}A "
        f"temp={status['temperature']:.0f}C "
        f"err={status['error']}"
    )


def move_to_target(adapter: MotorAdapter, args: argparse.Namespace) -> dict:
    deadline = time.monotonic() + args.move_timeout_s
    next_command = 0.0
    next_print = 0.0
    last_status = None

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_command:
            adapter.send_position_speed(args.target_deg, args.speed_erpm, args.rpa)
            next_command = now + MOVE_COMMAND_PERIOD_S

        status = adapter.get_status()
        if status is not None:
            last_status = status
            if now >= next_print:
                print_status("[move]", status, target_deg=args.target_deg)
                next_print = now + STATUS_PRINT_PERIOD_S

            if (
                abs(status["position"] - args.target_deg) <= args.position_tolerance_deg
                and abs(status["speed"]) <= args.speed_tolerance_erpm
            ):
                return status
        time.sleep(0.02)

    raise RuntimeError(f"Timed out before reaching target. Last status: {last_status}")


def hold_servo_position(adapter: MotorAdapter, args: argparse.Namespace) -> None:
    end_time = None if args.hold_seconds <= 0 else time.monotonic() + args.hold_seconds
    next_command = 0.0
    next_print = 0.0
    command_period_s = max(0.005, args.hold_command_period_s)

    if args.hold_mode == "pos":
        print(
            f"[hold] servo position hold via control mode 4 at {args.target_deg:.1f}deg. "
            "Press Ctrl+C to stop."
        )
    else:
        print(
            f"[hold] servo position-speed hold via control mode 6 at {args.target_deg:.1f}deg "
            f"(speed={args.hold_speed_erpm}, rpa={args.hold_rpa}). Press Ctrl+C to stop."
        )

    while end_time is None or time.monotonic() < end_time:
        now = time.monotonic()
        if now >= next_command:
            if args.hold_mode == "pos":
                adapter.send_position_only(args.target_deg)
            else:
                adapter.send_position_speed(args.target_deg, args.hold_speed_erpm, args.hold_rpa)
            next_command = now + command_period_s

        status = adapter.get_status()
        if status is not None and now >= next_print:
            print_status("[hold]", status, target_deg=args.target_deg)
            next_print = now + STATUS_PRINT_PERIOD_S
        time.sleep(0.02)


def main() -> int:
    args = parse_args()
    ser = None
    adapter = None

    try:
        ser = serial.Serial(
            args.port,
            CAN_BAUDRATE,
            timeout=SERIAL_TIMEOUT,
            write_timeout=WRITE_TIMEOUT,
        )
        time.sleep(0.1)
        ser.write(build_bridge_init_frame())
        time.sleep(0.1)

        adapter = MotorAdapter(args.family, ser, args.motor_id)
        status = try_wait_for_status(adapter, timeout_s=1.0)
        if status is not None:
            print_status("[ready]", status)
        else:
            print(
                "[info] no initial telemetry yet; proceeding to active command phase. "
                "Some AK70 firmwares only report after the first command."
            )

        if args.set_zero:
            print("[zero] setting current position to zero.")
            adapter.set_zero()
            time.sleep(0.5)
            status = try_wait_for_status(adapter, timeout_s=1.0)
            if status is not None:
                print_status("[zero]", status)
            else:
                print("[info] zero command sent; telemetry still not available yet.")

        reached = move_to_target(adapter, args)
        print_status("[reached]", reached, target_deg=args.target_deg)
        hold_servo_position(adapter, args)
        return 0
    except KeyboardInterrupt:
        print("\n[stop] interrupted by user.")
        return 130
    except Exception as exc:
        print(f"[error] {exc}")
        return 1
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass
        if ser is not None and ser.is_open:
            ser.close()


if __name__ == "__main__":
    raise SystemExit(main())
