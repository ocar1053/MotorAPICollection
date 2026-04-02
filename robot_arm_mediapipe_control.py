from __future__ import annotations

"""Real-time arm-follow controller using webcam(s) + MediaPipe Pose.

Default mapping for the clarified robot arm:
- joint_a: shoulder pitch, driven from landmark 11 area (shoulder pitch)
- joint_b: shoulder yaw, also driven from landmark 11 area (yaw)
- joint_c: elbow yaw, driven from landmark 13 area (relative forearm yaw)

Controls:
- q / ESC: quit
- c: recalibrate the current human pose as robot neutral
"""

import argparse
import math
import sys
import time
import warnings
from dataclasses import dataclass, replace

import cv2
import mediapipe as mp
import numpy as np

from robot_arm_runtime import (
    ArmController,
    DEFAULT_MOTORS,
    DEFAULT_PORTS,
    DEFAULT_TEMP_LIMIT_C,
    MotorSpec,
    clamp,
)

warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

if not hasattr(mp, "solutions"):
    raise SystemExit(
        "This script targets mediapipe==0.10.14. "
        "Please activate the environment where that version is installed."
    )

mp_pose = mp.solutions.pose


@dataclass
class RawArmPose:
    shoulder_pitch_deg: float
    shoulder_yaw_deg: float
    elbow_yaw_deg: float
    score: float


@dataclass
class Calibration:
    shoulder_pitch_deg: float
    shoulder_yaw_deg: float
    elbow_yaw_deg: float


@dataclass
class JointTargets:
    joint_a: float
    joint_b: float
    joint_c: float


class ExponentialSmoother:
    def __init__(self, alpha: float):
        self.alpha = clamp(alpha, 0.0, 1.0)
        self.value: JointTargets | None = None

    def update(self, target: JointTargets) -> JointTargets:
        if self.value is None or self.alpha <= 0.0:
            self.value = target
            return target

        self.value = JointTargets(
            joint_a=self.value.joint_a + (target.joint_a - self.value.joint_a) * self.alpha,
            joint_b=self.value.joint_b + (target.joint_b - self.value.joint_b) * self.alpha,
            joint_c=self.value.joint_c + (target.joint_c - self.value.joint_c) * self.alpha,
        )
        return self.value


class JointRateLimiter:
    def __init__(self, joint_a_max_deg_s: float, joint_b_max_deg_s: float, joint_c_max_deg_s: float):
        self.max_rates = JointTargets(
            joint_a=max(0.0, joint_a_max_deg_s),
            joint_b=max(0.0, joint_b_max_deg_s),
            joint_c=max(0.0, joint_c_max_deg_s),
        )
        self.value: JointTargets | None = None
        self._last_ts: float | None = None

    @staticmethod
    def _limit_axis(current: float, target: float, max_rate_deg_s: float, dt_s: float) -> float:
        if max_rate_deg_s <= 0.0:
            return current
        max_delta = max_rate_deg_s * dt_s
        return current + clamp(target - current, -max_delta, max_delta)

    def update(self, target: JointTargets, now_ts: float | None = None) -> JointTargets:
        now_ts = time.time() if now_ts is None else now_ts
        if self.value is None or self._last_ts is None:
            self.value = target
            self._last_ts = now_ts
            return target

        dt_s = max(1e-3, now_ts - self._last_ts)
        self.value = JointTargets(
            joint_a=self._limit_axis(self.value.joint_a, target.joint_a, self.max_rates.joint_a, dt_s),
            joint_b=self._limit_axis(self.value.joint_b, target.joint_b, self.max_rates.joint_b, dt_s),
            joint_c=self._limit_axis(self.value.joint_c, target.joint_c, self.max_rates.joint_c, dt_s),
        )
        self._last_ts = now_ts
        return self.value


class JointPIDController:
    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = max(0.0, integral_limit)
        self.integral = JointTargets(0.0, 0.0, 0.0)
        self.prev_error: JointTargets | None = None
        self.output: JointTargets | None = None
        self._last_ts: float | None = None

    def reset(self, output: JointTargets | None = None) -> None:
        self.integral = JointTargets(0.0, 0.0, 0.0)
        self.prev_error = None
        self.output = output
        self._last_ts = None

    @staticmethod
    def _axis(
        target: float,
        current: float,
        prev_error: float | None,
        integral: float,
        kp: float,
        ki: float,
        kd: float,
        dt_s: float,
        integral_limit: float,
    ) -> tuple[float, float, float]:
        error = target - current
        integral = clamp(integral + error * dt_s, -integral_limit, integral_limit)
        derivative = 0.0 if prev_error is None else (error - prev_error) / max(dt_s, 1e-3)
        output = current + kp * error + ki * integral + kd * derivative
        return output, integral, error

    def update(
        self,
        target: JointTargets,
        measured: JointTargets | None = None,
        now_ts: float | None = None,
    ) -> JointTargets:
        now_ts = time.time() if now_ts is None else now_ts
        current = measured or self.output or target
        if self.output is None:
            self.output = current
        if self._last_ts is None:
            self._last_ts = now_ts
            self.output = current
            return current

        dt_s = max(1e-3, now_ts - self._last_ts)

        joint_a, int_a, err_a = self._axis(
            target.joint_a,
            current.joint_a,
            None if self.prev_error is None else self.prev_error.joint_a,
            self.integral.joint_a,
            self.kp,
            self.ki,
            self.kd,
            dt_s,
            self.integral_limit,
        )
        joint_b, int_b, err_b = self._axis(
            target.joint_b,
            current.joint_b,
            None if self.prev_error is None else self.prev_error.joint_b,
            self.integral.joint_b,
            self.kp,
            self.ki,
            self.kd,
            dt_s,
            self.integral_limit,
        )
        joint_c, int_c, err_c = self._axis(
            target.joint_c,
            current.joint_c,
            None if self.prev_error is None else self.prev_error.joint_c,
            self.integral.joint_c,
            self.kp,
            self.ki,
            self.kd,
            dt_s,
            self.integral_limit,
        )

        self.integral = JointTargets(int_a, int_b, int_c)
        self.prev_error = JointTargets(err_a, err_b, err_c)
        self.output = JointTargets(joint_a, joint_b, joint_c)
        self._last_ts = now_ts
        return self.output


class EMAJointFilter:
    def __init__(self, alpha: float = 0.4):
        self.alpha = clamp(alpha, 0.0, 1.0)
        self.state: JointTargets | None = None

    def update(self, target: JointTargets) -> JointTargets:
        if self.state is None:
            self.state = target
            return target

        self.state = JointTargets(
            joint_a=self.alpha * target.joint_a + (1.0 - self.alpha) * self.state.joint_a,
            joint_b=self.alpha * target.joint_b + (1.0 - self.alpha) * self.state.joint_b,
            joint_c=self.alpha * target.joint_c + (1.0 - self.alpha) * self.state.joint_c,
        )
        return self.state


class OneEuroJointFilter:
    """Adapted from aurorachung0327/Gripper_Skeleton filter.py for scalar joint targets."""

    def __init__(self, min_cutoff: float = 1.7, beta: float = 0.5, d_cutoff: float = 1.0, freq: float = 30.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = max(freq, 1e-5)
        self.x_prev: JointTargets | None = None
        self.dx_prev: JointTargets | None = None
        self._last_ts: float | None = None

    def set_freq(self, freq: float) -> None:
        self.freq = max(freq, 1e-5)

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def _filter_axis(self, value: float, prev_value: float, prev_dx: float) -> tuple[float, float]:
        dx = (value - prev_value) * self.freq
        alpha_d = self._alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * prev_dx
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff)
        x_hat = alpha * value + (1.0 - alpha) * prev_value
        return x_hat, dx_hat

    def update(self, target: JointTargets, now_ts: float | None = None) -> JointTargets:
        now_ts = time.time() if now_ts is None else now_ts
        if self._last_ts is not None:
            dt_s = max(1e-3, now_ts - self._last_ts)
            self.set_freq(1.0 / dt_s)
        self._last_ts = now_ts

        if self.x_prev is None or self.dx_prev is None:
            self.x_prev = target
            self.dx_prev = JointTargets(0.0, 0.0, 0.0)
            return target

        joint_a, dx_a = self._filter_axis(target.joint_a, self.x_prev.joint_a, self.dx_prev.joint_a)
        joint_b, dx_b = self._filter_axis(target.joint_b, self.x_prev.joint_b, self.dx_prev.joint_b)
        joint_c, dx_c = self._filter_axis(target.joint_c, self.x_prev.joint_c, self.dx_prev.joint_c)

        self.x_prev = JointTargets(joint_a, joint_b, joint_c)
        self.dx_prev = JointTargets(dx_a, dx_b, dx_c)
        return self.x_prev


class CameraTracker:
    def __init__(
        self,
        camera_index: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ):
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera {camera_index}.")

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.capture.read()
        if not ok:
            return False, None
        return True, frame

    def close(self) -> None:
        try:
            self.pose.close()
        except Exception:
            pass
        try:
            self.capture.release()
        except Exception:
            pass


def normalize_angle_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def angle_delta_deg(current_deg: float, reference_deg: float) -> float:
    return normalize_angle_deg(current_deg - reference_deg)


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    magnitude = norm(vector)
    if magnitude < 1e-6:
        raise ValueError("Zero-length vector.")
    return vector / magnitude


def landmark_xyz(landmark) -> np.ndarray:
    return np.array([landmark.x, -landmark.y, -landmark.z], dtype=np.float64)


def selected_arm_indices(side: str) -> tuple[int, int, int]:
    if side == "left":
        return (
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value,
        )
    return (
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
    )


def project_body_frame(vector: np.ndarray, right_axis: np.ndarray, up_axis: np.ndarray, forward_axis: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(np.dot(vector, right_axis)),
            float(np.dot(vector, up_axis)),
            float(np.dot(vector, forward_axis)),
        ],
        dtype=np.float64,
    )


def body_yaw_deg(vector_body: np.ndarray) -> float:
    forward = float(vector_body[2])
    if abs(forward) < 1e-6:
        forward = 1e-6 if forward >= 0.0 else -1e-6
    return math.degrees(math.atan2(float(vector_body[0]), forward))


def body_pitch_deg(vector_body: np.ndarray) -> float:
    horizontal = max(1e-6, math.hypot(float(vector_body[0]), float(vector_body[2])))
    return math.degrees(math.atan2(float(vector_body[1]), horizontal))


def signed_horizontal_angle_deg(reference_body: np.ndarray, target_body: np.ndarray) -> float:
    reference = np.array([float(reference_body[0]), float(reference_body[2])], dtype=np.float64)
    target = np.array([float(target_body[0]), float(target_body[2])], dtype=np.float64)
    reference_norm = float(np.linalg.norm(reference))
    target_norm = float(np.linalg.norm(target))
    if reference_norm < 1e-6 or target_norm < 1e-6:
        return 0.0
    reference /= reference_norm
    target /= target_norm
    determinant = reference[0] * target[1] - reference[1] * target[0]
    dot = clamp(float(np.dot(reference, target)), -1.0, 1.0)
    return math.degrees(math.atan2(determinant, dot))


def extract_arm_pose(
    results,
    side: str,
    min_visibility: float,
) -> RawArmPose | None:
    if not results.pose_landmarks:
        return None

    normalized = results.pose_landmarks.landmark
    world = results.pose_world_landmarks.landmark if results.pose_world_landmarks else normalized

    shoulder_index, elbow_index, wrist_index = selected_arm_indices(side)

    required_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        shoulder_index,
        elbow_index,
        wrist_index,
    ]
    if any(normalized[index].visibility < min_visibility for index in required_indices):
        return None

    left_shoulder = landmark_xyz(world[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    right_shoulder = landmark_xyz(world[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    left_hip = landmark_xyz(world[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_hip = landmark_xyz(world[mp_pose.PoseLandmark.RIGHT_HIP.value])
    shoulder = landmark_xyz(world[shoulder_index])
    elbow = landmark_xyz(world[elbow_index])
    wrist = landmark_xyz(world[wrist_index])

    shoulder_center = (left_shoulder + right_shoulder) * 0.5
    right_axis = normalize_vector(right_shoulder - left_shoulder)
    hip_visible = (
        normalized[mp_pose.PoseLandmark.LEFT_HIP.value].visibility >= min_visibility
        and normalized[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility >= min_visibility
    )
    if hip_visible:
        hip_center = (left_hip + right_hip) * 0.5
        tentative_up = shoulder_center - hip_center
    else:
        tentative_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    if norm(tentative_up) < 1e-6:
        tentative_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    forward_candidate = np.cross(right_axis, tentative_up)
    if norm(forward_candidate) < 1e-6:
        forward_candidate = np.cross(right_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    forward_axis = normalize_vector(forward_candidate)
    up_axis = normalize_vector(np.cross(forward_axis, right_axis))

    upper_arm = elbow - shoulder
    forearm = wrist - elbow
    if norm(upper_arm) < 1e-4 or norm(forearm) < 1e-4:
        return None

    upper_arm_body = project_body_frame(upper_arm, right_axis, up_axis, forward_axis)
    forearm_body = project_body_frame(forearm, right_axis, up_axis, forward_axis)

    shoulder_pitch_deg = body_pitch_deg(upper_arm_body)
    shoulder_yaw_deg = body_yaw_deg(upper_arm_body)
    elbow_yaw_deg = signed_horizontal_angle_deg(upper_arm_body, forearm_body)

    score = float(
        (
            normalized[shoulder_index].visibility
            + normalized[elbow_index].visibility
            + normalized[wrist_index].visibility
        )
        / 3.0
    )

    return RawArmPose(
        shoulder_pitch_deg=shoulder_pitch_deg,
        shoulder_yaw_deg=shoulder_yaw_deg,
        elbow_yaw_deg=elbow_yaw_deg,
        score=score,
    )


def compose_grid(frames: list[np.ndarray], cell_width: int = 640, cell_height: int = 360) -> np.ndarray:
    prepared = [cv2.resize(frame, (cell_width, cell_height)) for frame in frames]
    if len(prepared) == 1:
        return prepared[0]

    cols = 2
    rows = math.ceil(len(prepared) / cols)
    blank = np.zeros_like(prepared[0])
    while len(prepared) < rows * cols:
        prepared.append(blank.copy())

    strips = []
    for row_index in range(rows):
        start = row_index * cols
        strips.append(np.hstack(prepared[start:start + cols]))
    return np.vstack(strips)


def initial_window_size(num_frames: int, cell_width: int, cell_height: int) -> tuple[int, int]:
    if num_frames <= 1:
        return cell_width, cell_height
    cols = 2
    rows = math.ceil(num_frames / cols)
    return cell_width * cols, cell_height * rows


def draw_arm_skeleton(
    frame: np.ndarray,
    results,
    side: str,
    min_visibility: float,
    mirror: bool,
    active: bool,
) -> np.ndarray:
    if not results.pose_landmarks:
        return frame

    landmarks = results.pose_landmarks.landmark
    shoulder_index, elbow_index, wrist_index = selected_arm_indices(side)
    arm_indices = (shoulder_index, elbow_index, wrist_index)
    if any(landmarks[index].visibility < min_visibility for index in arm_indices):
        return frame

    frame_height, frame_width = frame.shape[:2]
    points: list[tuple[int, int]] = []
    for index in arm_indices:
        landmark = landmarks[index]
        x = int(round(landmark.x * frame_width))
        y = int(round(landmark.y * frame_height))
        if mirror:
            x = frame_width - x
        points.append((x, y))

    line_color = (40, 200, 120) if active else (120, 120, 120)
    joint_color = (245, 245, 245)
    cv2.line(frame, points[0], points[1], line_color, 4, cv2.LINE_AA)
    cv2.line(frame, points[1], points[2], line_color, 4, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 8, joint_color, -1, cv2.LINE_AA)
        cv2.circle(frame, point, 10, line_color, 2, cv2.LINE_AA)
    return frame


def annotate_frame(
    frame: np.ndarray,
    camera_index: int,
    active: bool,
    raw_pose: RawArmPose | None,
    joint_targets: JointTargets | None,
    status_text: str,
) -> np.ndarray:
    color = (40, 200, 120) if active else (110, 110, 110)
    cv2.rectangle(frame, (6, 6), (frame.shape[1] - 6, frame.shape[0] - 6), color, 2)

    lines = [f"cam {camera_index}{' active' if active else ''}", status_text]
    if raw_pose is not None:
        lines.append(
            f"human pitch {raw_pose.shoulder_pitch_deg:+.1f} | shoulder yaw {raw_pose.shoulder_yaw_deg:+.1f} | elbow yaw {raw_pose.elbow_yaw_deg:+.1f}"
        )
    if joint_targets is not None:
        lines.append(
            f"joint_a {joint_targets.joint_a:+.1f} | joint_b {joint_targets.joint_b:+.1f} | joint_c {joint_targets.joint_c:+.1f}"
        )

    for line_index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 30 + line_index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    return frame


def print_joint_targets_live(
    active_camera_index: int | None,
    raw_pose: RawArmPose | None,
    joint_targets: JointTargets | None,
    arm_enabled: bool,
    side: str,
) -> None:
    if raw_pose is None or joint_targets is None or active_camera_index is None:
        message = f"[live] waiting for arm pose... keep your {side} arm and both shoulders in frame"
    else:
        mode = "ARM" if arm_enabled else "PREVIEW"
        message = (
            f"[live][cam {active_camera_index}][{mode}] "
            f"joint_a={joint_targets.joint_a:+06.1f}  "
            f"joint_b={joint_targets.joint_b:+06.1f}  "
            f"joint_c={joint_targets.joint_c:+06.1f}"
        )

    now = time.time()
    last_message = getattr(print_joint_targets_live, "_last_message", None)
    last_print_at = getattr(print_joint_targets_live, "_last_print_at", 0.0)
    if message != last_message or now - last_print_at >= 0.25:
        print(message, flush=True)
        print_joint_targets_live._last_message = message
        print_joint_targets_live._last_print_at = now


def add_joint_limit_args(parser: argparse.ArgumentParser, key: str, label: str) -> None:
    parser.add_argument(
        f"--{key.replace('_', '-')}-min",
        type=float,
        default=None,
        help=f"Override the minimum allowed angle for {label}.",
    )
    parser.add_argument(
        f"--{key.replace('_', '-')}-max",
        type=float,
        default=None,
        help=f"Override the maximum allowed angle for {label}.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Follow your arm with MediaPipe Pose and map it to joint_a/joint_b/joint_c.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        type=int,
        dest="cameras",
        help="Camera index. Pass multiple times to use more than one webcam. Default: 0",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="left",
        help="Which arm to follow. Default: left (MediaPipe 11/13/15 side).",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORTS["A"],
        help=f"USB-to-CAN COM port. Default: {DEFAULT_PORTS['A']}",
    )
    parser.add_argument(
        "--arm",
        action="store_true",
        help="Actually send commands to the robot arm. Without this flag, preview only.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the preview window for easier self-view. Tracking still uses the original frame.",
    )
    parser.add_argument(
        "--window-cell-width",
        type=int,
        default=1280,
        help="Display width for each camera cell in the preview window.",
    )
    parser.add_argument(
        "--window-cell-height",
        type=int,
        default=720,
        help="Display height for each camera cell in the preview window.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.65,
        help="MediaPipe detection confidence threshold.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.65,
        help="MediaPipe tracking confidence threshold.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.35,
        help="Minimum landmark visibility to accept a pose.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Exponential smoothing factor in [0, 1]. Higher follows faster.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("direct", "pid"),
        default="direct",
        help="`direct` sends filtered targets straight to the arm. `pid` adds an outer PID loop.",
    )
    parser.add_argument(
        "--temporal-filter",
        choices=("none", "ema", "oneeuro"),
        default="oneeuro",
        help="Temporal filter for joint targets. `oneeuro` is adapted from Gripper_Skeleton.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.35,
        help="EMA alpha when --temporal-filter ema is used.",
    )
    parser.add_argument(
        "--oneeuro-min-cutoff",
        type=float,
        default=1.1,
        help="One Euro filter minimum cutoff.",
    )
    parser.add_argument(
        "--oneeuro-beta",
        type=float,
        default=1.0,
        help="One Euro filter beta. Lower is smoother, higher is more responsive.",
    )
    parser.add_argument(
        "--oneeuro-d-cutoff",
        type=float,
        default=1.0,
        help="One Euro derivative cutoff.",
    )
    parser.add_argument(
        "--send-interval-ms",
        type=float,
        default=35.0,
        help="Minimum interval between command bursts. Set 0 to send every loop.",
    )
    parser.add_argument(
        "--joint-a-max-deg-s",
        type=float,
        default=0.0,
        help="Maximum speed for joint_a target changes in deg/s. Set 0 to disable software rate limiting.",
    )
    parser.add_argument(
        "--joint-b-max-deg-s",
        type=float,
        default=0.0,
        help="Maximum speed for joint_b target changes in deg/s. Set 0 to disable software rate limiting.",
    )
    parser.add_argument(
        "--joint-c-max-deg-s",
        type=float,
        default=0.0,
        help="Maximum speed for joint_c target changes in deg/s. Set 0 to disable software rate limiting.",
    )
    parser.add_argument(
        "--pid-kp",
        type=float,
        default=0.40,
        help="PID proportional gain for joint target tracking when --control-mode pid is used.",
    )
    parser.add_argument(
        "--pid-ki",
        type=float,
        default=0.02,
        help="PID integral gain for joint target tracking when --control-mode pid is used.",
    )
    parser.add_argument(
        "--pid-kd",
        type=float,
        default=0.05,
        help="PID derivative gain for joint target tracking when --control-mode pid is used.",
    )
    parser.add_argument(
        "--pid-integral-limit",
        type=float,
        default=25.0,
        help="Clamp integral accumulation to avoid PID wind-up.",
    )
    parser.add_argument(
        "--joint-a-gain",
        type=float,
        default=0.70,
        help="Scale from human shoulder pitch delta to joint_a delta.",
    )
    parser.add_argument(
        "--joint-b-gain",
        type=float,
        default=0.85,
        help="Scale from human shoulder yaw delta to joint_b delta.",
    )
    parser.add_argument(
        "--joint-c-gain",
        type=float,
        default=0.85,
        help="Scale from human elbow yaw delta to joint_c delta.",
    )
    parser.add_argument(
        "--joint-a-sign",
        type=float,
        default=-1.0,
        help="Direction multiplier for joint_a.",
    )
    parser.add_argument(
        "--joint-b-sign",
        type=float,
        default=None,
        help="Direction multiplier for joint_b. If omitted, right-arm mode flips yaw by default.",
    )
    parser.add_argument(
        "--joint-c-sign",
        type=float,
        default=None,
        help="Direction multiplier for joint_c. If omitted, right-arm mode flips yaw by default.",
    )
    parser.add_argument(
        "--joint-a-offset",
        type=float,
        default=0.0,
        help="Extra offset added to joint_a after calibration.",
    )
    parser.add_argument(
        "--joint-b-offset",
        type=float,
        default=0.0,
        help="Extra offset added to joint_b after calibration.",
    )
    parser.add_argument(
        "--joint-c-offset",
        type=float,
        default=0.0,
        help="Extra offset added to joint_c after calibration.",
    )
    add_joint_limit_args(parser, "joint_a", "joint_a")
    add_joint_limit_args(parser, "joint_b", "joint_b")
    add_joint_limit_args(parser, "joint_c", "joint_c")
    return parser.parse_args()


def resolve_side_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.joint_b_sign is None:
        args.joint_b_sign = -1.0 if args.side == "right" else 1.0
    if args.joint_c_sign is None:
        args.joint_c_sign = -1.0 if args.side == "right" else 1.0
    return args


def safety_check(controller: ArmController) -> None:
    states = controller.refresh_states()
    for key, state in states.items():
        if state.error:
            raise RuntimeError(f"{key} reported motor error code {state.error}.")
        if state.temperature >= DEFAULT_TEMP_LIMIT_C:
            raise RuntimeError(
                f"{key} temperature {state.temperature:.1f} C exceeded the safe limit {DEFAULT_TEMP_LIMIT_C} C."
            )


def wait_for_initial_telemetry(controller: ArmController, timeout_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        states = controller.refresh_states()
        if all(state.bus_connected and state.telemetry_ok for state in states.values()):
            return
        time.sleep(0.05)
    missing = [
        key
        for key, state in controller.refresh_states().items()
        if not (state.bus_connected and state.telemetry_ok)
    ]
    raise RuntimeError(f"Telemetry did not become ready for: {', '.join(missing)}")


def current_robot_home(controller: ArmController) -> JointTargets:
    states = controller.refresh_states()
    return JointTargets(
        joint_a=states["joint_a"].position,
        joint_b=states["joint_b"].position,
        joint_c=states["joint_c"].position,
    )


def current_joint_measurement(controller: ArmController | None) -> JointTargets | None:
    if controller is None:
        return None
    return JointTargets(
        joint_a=controller.states["joint_a"].position,
        joint_b=controller.states["joint_b"].position,
        joint_c=controller.states["joint_c"].position,
    )


def build_motor_specs(args: argparse.Namespace) -> list[MotorSpec]:
    specs: list[MotorSpec] = []
    for spec in DEFAULT_MOTORS:
        min_override = getattr(args, f"{spec.key}_min")
        max_override = getattr(args, f"{spec.key}_max")
        min_deg = spec.min_deg if min_override is None else min_override
        max_deg = spec.max_deg if max_override is None else max_override
        if min_deg > max_deg:
            raise SystemExit(f"{spec.key}: min limit {min_deg} cannot be greater than max limit {max_deg}.")
        specs.append(replace(spec, min_deg=min_deg, max_deg=max_deg))
    return specs


def specs_by_key(specs: list[MotorSpec]) -> dict[str, MotorSpec]:
    return {spec.key: spec for spec in specs}


def raw_to_joint_targets(
    raw_pose: RawArmPose,
    calibration: Calibration,
    home_targets: JointTargets,
    limits_by_key: dict[str, MotorSpec],
    args: argparse.Namespace,
) -> JointTargets:
    pitch_delta = angle_delta_deg(raw_pose.shoulder_pitch_deg, calibration.shoulder_pitch_deg)
    shoulder_yaw_delta = angle_delta_deg(raw_pose.shoulder_yaw_deg, calibration.shoulder_yaw_deg)
    elbow_yaw_delta = angle_delta_deg(raw_pose.elbow_yaw_deg, calibration.elbow_yaw_deg)

    desired = JointTargets(
        joint_a=home_targets.joint_a + pitch_delta * args.joint_a_gain * args.joint_a_sign + args.joint_a_offset,
        joint_b=home_targets.joint_b + shoulder_yaw_delta * args.joint_b_gain * args.joint_b_sign + args.joint_b_offset,
        joint_c=home_targets.joint_c + elbow_yaw_delta * args.joint_c_gain * args.joint_c_sign + args.joint_c_offset,
    )

    return JointTargets(
        joint_a=clamp(desired.joint_a, limits_by_key["joint_a"].min_deg, limits_by_key["joint_a"].max_deg),
        joint_b=clamp(desired.joint_b, limits_by_key["joint_b"].min_deg, limits_by_key["joint_b"].max_deg),
        joint_c=clamp(desired.joint_c, limits_by_key["joint_c"].min_deg, limits_by_key["joint_c"].max_deg),
    )


def apply_joint_targets(controller: ArmController, targets: JointTargets) -> None:
    controller.set_target("joint_a", targets.joint_a)
    controller.set_target("joint_b", targets.joint_b)
    controller.set_target("joint_c", targets.joint_c)
    controller.command_motor("joint_a", force=True)
    controller.command_motor("joint_b", force=True)
    controller.command_motor("joint_c", force=True)


def initialize_controller(port: str, specs: list[MotorSpec]) -> ArmController:
    controller = ArmController(specs)
    controller.connect_buses({"A": port})
    wait_for_initial_telemetry(controller)
    controller.set_motion_armed(True)
    safety_check(controller)
    return controller


def build_temporal_filter(args: argparse.Namespace):
    if args.temporal_filter == "none":
        return None
    if args.temporal_filter == "ema":
        return EMAJointFilter(alpha=args.ema_alpha)
    return OneEuroJointFilter(
        min_cutoff=args.oneeuro_min_cutoff,
        beta=args.oneeuro_beta,
        d_cutoff=args.oneeuro_d_cutoff,
    )


def main() -> int:
    args = resolve_side_defaults(parse_args())
    cameras = args.cameras or [0]
    motor_specs = build_motor_specs(args)
    limits_by_key = specs_by_key(motor_specs)
    window_name = "Robot Arm MediaPipe Control"
    trackers: list[CameraTracker] = []
    controller: ArmController | None = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    initial_width, initial_height = initial_window_size(
        num_frames=len(cameras),
        cell_width=max(320, args.window_cell_width),
        cell_height=max(240, args.window_cell_height),
    )
    cv2.resizeWindow(window_name, initial_width, initial_height)

    try:
        trackers = [
            CameraTracker(
                camera_index=camera_index,
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
            )
            for camera_index in cameras
        ]
    except Exception as exc:
        for tracker in trackers:
            tracker.close()
        print(f"[camera] {exc}", file=sys.stderr)
        return 1

    try:
        if args.arm:
            controller = initialize_controller(args.port, motor_specs)
            home_targets = current_robot_home(controller)
            print(
                "[robot] armed on "
                f"{args.port} with home joint targets "
                f"A={home_targets.joint_a:.1f}, B={home_targets.joint_b:.1f}, C={home_targets.joint_c:.1f}"
            )
        else:
            home_targets = JointTargets(0.0, 0.0, 0.0)
            print("[robot] preview only. Add --arm to actually move the robot.")

        print(
            "[limits] "
            f"joint_a=[{limits_by_key['joint_a'].min_deg:.1f}, {limits_by_key['joint_a'].max_deg:.1f}] "
            f"joint_b=[{limits_by_key['joint_b'].min_deg:.1f}, {limits_by_key['joint_b'].max_deg:.1f}] "
            f"joint_c=[{limits_by_key['joint_c'].min_deg:.1f}, {limits_by_key['joint_c'].max_deg:.1f}]"
        )
        print(
            "[control] "
            f"mode={args.control_mode} "
            f"send_interval_ms={args.send_interval_ms:.0f}"
        )
        if any(rate > 0.0 for rate in (args.joint_a_max_deg_s, args.joint_b_max_deg_s, args.joint_c_max_deg_s)):
            print(
                "[rate-limit] "
                f"joint_a={args.joint_a_max_deg_s:.1f} deg/s "
                f"joint_b={args.joint_b_max_deg_s:.1f} deg/s "
                f"joint_c={args.joint_c_max_deg_s:.1f} deg/s"
            )
        else:
            print("[rate-limit] disabled")
        print(
            "[filter] "
            f"type={args.temporal_filter} "
            f"ema_alpha={args.ema_alpha:.2f} "
            f"oneeuro_min_cutoff={args.oneeuro_min_cutoff:.2f} "
            f"oneeuro_beta={args.oneeuro_beta:.2f}"
        )
        if args.control_mode == "pid":
            print(
                "[pid] "
                f"kp={args.pid_kp:.3f} "
                f"ki={args.pid_ki:.3f} "
                f"kd={args.pid_kd:.3f} "
                f"integral_limit={args.pid_integral_limit:.1f}"
            )
        print(
            "[mapping] "
            f"side={args.side} "
            f"joint_a_sign={args.joint_a_sign:+.1f} "
            f"joint_b_sign={args.joint_b_sign:+.1f} "
            f"joint_c_sign={args.joint_c_sign:+.1f}"
        )

        print("[input] press c to recalibrate neutral pose, q to quit.")

        smoother = ExponentialSmoother(alpha=args.smoothing)
        temporal_filter = build_temporal_filter(args)
        pid_controller = (
            JointPIDController(
                kp=args.pid_kp,
                ki=args.pid_ki,
                kd=args.pid_kd,
                integral_limit=args.pid_integral_limit,
            )
            if args.control_mode == "pid"
            else None
        )
        rate_limiter = (
            JointRateLimiter(
                joint_a_max_deg_s=args.joint_a_max_deg_s,
                joint_b_max_deg_s=args.joint_b_max_deg_s,
                joint_c_max_deg_s=args.joint_c_max_deg_s,
            )
            if any(rate > 0.0 for rate in (args.joint_a_max_deg_s, args.joint_b_max_deg_s, args.joint_c_max_deg_s))
            else None
        )
        calibration: Calibration | None = None
        last_sent_at = 0.0
        last_poll_at = 0.0
        send_interval_s = max(0.0, args.send_interval_ms / 1000.0)

        while True:
            best_camera_index: int | None = None
            best_raw_pose: RawArmPose | None = None
            best_joint_targets: JointTargets | None = None
            frames: list[np.ndarray] = []
            raw_pose_by_camera: dict[int, RawArmPose | None] = {}

            for tracker in trackers:
                ok, frame = tracker.read()
                if not ok or frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        f"camera {tracker.camera_index} read failed",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    frames.append(frame)
                    raw_pose_by_camera[tracker.camera_index] = None
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = tracker.pose.process(rgb)
                display_frame = frame.copy()

                raw_pose = extract_arm_pose(
                    results=results,
                    side=args.side,
                    min_visibility=args.min_visibility,
                )
                raw_pose_by_camera[tracker.camera_index] = raw_pose

                if raw_pose is not None and (best_raw_pose is None or raw_pose.score > best_raw_pose.score):
                    best_camera_index = tracker.camera_index
                    best_raw_pose = raw_pose

                if args.mirror:
                    display_frame = cv2.flip(display_frame, 1)

                display_frame = draw_arm_skeleton(
                    frame=display_frame,
                    results=results,
                    side=args.side,
                    min_visibility=args.min_visibility,
                    mirror=args.mirror,
                    active=tracker.camera_index == best_camera_index,
                )

                frames.append(display_frame)

            if best_raw_pose is not None and calibration is None:
                calibration = Calibration(
                    shoulder_pitch_deg=best_raw_pose.shoulder_pitch_deg,
                    shoulder_yaw_deg=best_raw_pose.shoulder_yaw_deg,
                    elbow_yaw_deg=best_raw_pose.elbow_yaw_deg,
                )
                if pid_controller is not None:
                    pid_controller.reset(output=current_joint_measurement(controller) or home_targets)
                print()
                print(
                    "[calibration] captured neutral pose: "
                    f"pitch={calibration.shoulder_pitch_deg:+.1f}, "
                    f"shoulder_yaw={calibration.shoulder_yaw_deg:+.1f}, "
                    f"elbow_yaw={calibration.elbow_yaw_deg:+.1f}"
                )

            if best_raw_pose is not None and calibration is not None:
                raw_targets = raw_to_joint_targets(
                    raw_pose=best_raw_pose,
                    calibration=calibration,
                    home_targets=home_targets,
                    limits_by_key=limits_by_key,
                    args=args,
                )
                smoothed_targets = smoother.update(raw_targets)
                if temporal_filter is not None:
                    if isinstance(temporal_filter, OneEuroJointFilter):
                        smoothed_targets = temporal_filter.update(smoothed_targets, now_ts=time.time())
                    else:
                        smoothed_targets = temporal_filter.update(smoothed_targets)

                control_targets = smoothed_targets
                if pid_controller is not None:
                    control_targets = pid_controller.update(
                        smoothed_targets,
                        measured=current_joint_measurement(controller),
                        now_ts=time.time(),
                    )
                if rate_limiter is not None:
                    control_targets = rate_limiter.update(control_targets, now_ts=time.time())
                best_joint_targets = control_targets

            print_joint_targets_live(
                active_camera_index=best_camera_index,
                raw_pose=best_raw_pose,
                joint_targets=best_joint_targets,
                arm_enabled=args.arm,
                side=args.side,
            )

            now = time.time()
            if controller is not None:
                if now - last_poll_at >= 0.15:
                    safety_check(controller)
                    for key in ("joint_a", "joint_b", "joint_c"):
                        controller.try_auto_hold(key)
                    last_poll_at = now

                controller.maintain_target_holds()

                if best_joint_targets is not None and (
                    send_interval_s <= 0.0 or now - last_sent_at >= send_interval_s
                ):
                    apply_joint_targets(controller, best_joint_targets)
                    last_sent_at = now

            annotated_frames: list[np.ndarray] = []
            for tracker, frame in zip(trackers, frames):
                raw_pose = raw_pose_by_camera.get(tracker.camera_index)
                active = tracker.camera_index == best_camera_index
                if raw_pose is None:
                    status_text = "pose not locked"
                elif calibration is None:
                    status_text = "waiting for calibration"
                elif active:
                    status_text = "following"
                else:
                    status_text = "tracked"

                annotated_frames.append(
                    annotate_frame(
                        frame=frame,
                        camera_index=tracker.camera_index,
                        active=active,
                        raw_pose=raw_pose,
                        joint_targets=best_joint_targets if active else None,
                        status_text=status_text,
                    )
                )

            canvas = compose_grid(
                annotated_frames,
                cell_width=max(320, args.window_cell_width),
                cell_height=max(240, args.window_cell_height),
            )
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c") and best_raw_pose is not None:
                calibration = Calibration(
                    shoulder_pitch_deg=best_raw_pose.shoulder_pitch_deg,
                    shoulder_yaw_deg=best_raw_pose.shoulder_yaw_deg,
                    elbow_yaw_deg=best_raw_pose.elbow_yaw_deg,
                )
                if controller is not None:
                    home_targets = current_robot_home(controller)
                if pid_controller is not None:
                    pid_controller.reset(output=current_joint_measurement(controller) or home_targets)
                print()
                print(
                    "[calibration] reset neutral pose: "
                    f"pitch={calibration.shoulder_pitch_deg:+.1f}, "
                    f"shoulder_yaw={calibration.shoulder_yaw_deg:+.1f}, "
                    f"elbow_yaw={calibration.elbow_yaw_deg:+.1f}"
                )

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        print()
        if controller is not None:
            try:
                controller.hold_all()
            except Exception:
                pass
            controller.disconnect_all()
        for tracker in trackers:
            tracker.close()
        cv2.destroyAllWindows()

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
