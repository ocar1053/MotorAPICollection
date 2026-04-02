from __future__ import annotations

"""RealSense D455 + MediaPipe arm-follow controller.

This script keeps the webcam-based controller separate and uses aligned depth
from a RealSense D455 to improve the 3D estimate of shoulder/elbow/wrist,
especially for joint_c / motor ID 93.
"""

import argparse
import math
import sys
import time
import warnings
from dataclasses import replace

import cv2
import mediapipe as mp
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "pyrealsense2 is required for the D455 script. "
        "Install it in this environment first: pip install pyrealsense2"
    ) from exc

from robot_arm_mediapipe_control import (
    Calibration,
    EMAJointFilter,
    ExponentialSmoother,
    JointPIDController,
    JointRateLimiter,
    JointTargets,
    OneEuroJointFilter,
    RawArmPose,
    annotate_frame,
    body_pitch_deg,
    body_yaw_deg,
    build_motor_specs,
    current_joint_measurement,
    current_robot_home,
    draw_arm_skeleton,
    initialize_controller,
    normalize_vector,
    norm,
    print_joint_targets_live,
    project_body_frame,
    raw_to_joint_targets,
    resolve_side_defaults,
    safety_check,
    selected_arm_indices,
    signed_horizontal_angle_deg,
)
from robot_arm_runtime import DEFAULT_PORTS, DEFAULT_TEMP_LIMIT_C

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


class RealSenseTracker:
    def __init__(
        self,
        color_width: int,
        color_height: int,
        fps: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        serial_number: str | None = None,
    ):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial_number:
            self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.depth, color_width, color_height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _ensure_depth_frame(frame) -> rs.depth_frame | None:
        if frame is None:
            return None
        try:
            depth_frame = frame.as_depth_frame()
        except AttributeError:
            depth_frame = frame
        if depth_frame is None:
            return None
        try:
            depth_frame.get_distance(0, 0)
        except Exception:
            return None
        return depth_frame

    def read(self) -> tuple[bool, np.ndarray | None, rs.depth_frame | None]:
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            return False, None, None

        aligned = self.align.process(frames)
        depth_frame = self._ensure_depth_frame(aligned.get_depth_frame())
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None, None

        depth_frame = self._ensure_depth_frame(self.spatial_filter.process(depth_frame))
        depth_frame = self._ensure_depth_frame(self.temporal_filter.process(depth_frame))
        depth_frame = self._ensure_depth_frame(self.hole_filling_filter.process(depth_frame))
        if depth_frame is None:
            return False, None, None
        color_image = np.asanyarray(color_frame.get_data())
        return True, color_image, depth_frame

    def close(self) -> None:
        try:
            self.pose.close()
        except Exception:
            pass
        try:
            self.pipeline.stop()
        except Exception:
            pass


def sample_depth_m(
    depth_frame: rs.depth_frame,
    x_px: int,
    y_px: int,
    radius_px: int,
    depth_min_m: float,
    depth_max_m: float,
) -> float | None:
    profile = depth_frame.profile.as_video_stream_profile()
    width = int(profile.width())
    height = int(profile.height())
    x0 = max(0, x_px - radius_px)
    x1 = min(width - 1, x_px + radius_px)
    y0 = max(0, y_px - radius_px)
    y1 = min(height - 1, y_px + radius_px)

    samples: list[float] = []
    for py in range(y0, y1 + 1):
        for px in range(x0, x1 + 1):
            distance_m = float(depth_frame.get_distance(px, py))
            if depth_min_m <= distance_m <= depth_max_m:
                samples.append(distance_m)

    if not samples:
        return None
    return float(np.median(samples))


def deproject_landmark_to_camera_xyz(
    landmark,
    depth_frame: rs.depth_frame,
    intrinsics,
    radius_px: int,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray | None:
    profile = depth_frame.profile.as_video_stream_profile()
    width = int(profile.width())
    height = int(profile.height())
    x_px = int(round(landmark.x * (width - 1)))
    y_px = int(round(landmark.y * (height - 1)))
    depth_m = sample_depth_m(
        depth_frame=depth_frame,
        x_px=x_px,
        y_px=y_px,
        radius_px=radius_px,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    if depth_m is None:
        return None
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(x_px), float(y_px)], depth_m)
    return np.array([point[0], -point[1], -point[2]], dtype=np.float64)


def horizontal_projection_ratio(vector_body: np.ndarray) -> float:
    total = max(1e-6, norm(vector_body))
    horizontal = float(np.hypot(float(vector_body[0]), float(vector_body[2])))
    return horizontal / total


def guarded_body_yaw_deg(
    vector_body: np.ndarray,
    min_horizontal_ratio: float,
    zero_snap_deg: float,
) -> float:
    if horizontal_projection_ratio(vector_body) < min_horizontal_ratio:
        return 0.0
    angle = body_yaw_deg(vector_body)
    if abs(angle) <= zero_snap_deg:
        return 0.0
    return angle


def guarded_relative_yaw_deg(
    reference_body: np.ndarray,
    target_body: np.ndarray,
    min_horizontal_ratio: float,
    zero_snap_deg: float,
) -> float:
    if (
        horizontal_projection_ratio(reference_body) < min_horizontal_ratio
        or horizontal_projection_ratio(target_body) < min_horizontal_ratio
    ):
        return 0.0
    angle = signed_horizontal_angle_deg(reference_body, target_body)
    if abs(angle) <= zero_snap_deg:
        return 0.0
    return angle


def coronal_plane_deviation_deg(vector_body: np.ndarray) -> float:
    lateral_vertical_mag = max(1e-6, math.hypot(float(vector_body[0]), float(vector_body[1])))
    return math.degrees(math.atan2(float(vector_body[2]), lateral_vertical_mag))


def guarded_coronal_plane_deviation_deg(vector_body: np.ndarray, zero_snap_deg: float) -> float:
    angle = coronal_plane_deviation_deg(vector_body)
    if abs(angle) <= zero_snap_deg:
        return 0.0
    return angle


def extract_depth_arm_pose(
    results,
    depth_frame: rs.depth_frame,
    side: str,
    min_visibility: float,
    depth_radius_px: int,
    depth_min_m: float,
    depth_max_m: float,
    yaw_singularity_ratio: float,
    joint_b_zero_snap_deg: float,
    joint_c_zero_snap_deg: float,
) -> RawArmPose | None:
    if not results.pose_landmarks:
        return None

    normalized = results.pose_landmarks.landmark
    shoulder_index, elbow_index, wrist_index = selected_arm_indices(side)
    left_shoulder_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    right_shoulder_index = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    left_hip_index = mp_pose.PoseLandmark.LEFT_HIP.value
    right_hip_index = mp_pose.PoseLandmark.RIGHT_HIP.value

    required_indices = [
        left_shoulder_index,
        right_shoulder_index,
        shoulder_index,
        elbow_index,
        wrist_index,
    ]
    if any(normalized[index].visibility < min_visibility for index in required_indices):
        return None

    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    left_shoulder = deproject_landmark_to_camera_xyz(
        normalized[left_shoulder_index],
        depth_frame,
        intrinsics,
        depth_radius_px,
        depth_min_m,
        depth_max_m,
    )
    right_shoulder = deproject_landmark_to_camera_xyz(
        normalized[right_shoulder_index],
        depth_frame,
        intrinsics,
        depth_radius_px,
        depth_min_m,
        depth_max_m,
    )
    shoulder = deproject_landmark_to_camera_xyz(
        normalized[shoulder_index],
        depth_frame,
        intrinsics,
        depth_radius_px,
        depth_min_m,
        depth_max_m,
    )
    elbow = deproject_landmark_to_camera_xyz(
        normalized[elbow_index],
        depth_frame,
        intrinsics,
        depth_radius_px,
        depth_min_m,
        depth_max_m,
    )
    wrist = deproject_landmark_to_camera_xyz(
        normalized[wrist_index],
        depth_frame,
        intrinsics,
        depth_radius_px,
        depth_min_m,
        depth_max_m,
    )

    if any(point is None for point in (left_shoulder, right_shoulder, shoulder, elbow, wrist)):
        return None

    left_hip = None
    right_hip = None
    if (
        normalized[left_hip_index].visibility >= min_visibility
        and normalized[right_hip_index].visibility >= min_visibility
    ):
        left_hip = deproject_landmark_to_camera_xyz(
            normalized[left_hip_index],
            depth_frame,
            intrinsics,
            depth_radius_px,
            depth_min_m,
            depth_max_m,
        )
        right_hip = deproject_landmark_to_camera_xyz(
            normalized[right_hip_index],
            depth_frame,
            intrinsics,
            depth_radius_px,
            depth_min_m,
            depth_max_m,
        )

    shoulder_center = (left_shoulder + right_shoulder) * 0.5
    right_axis = normalize_vector(right_shoulder - left_shoulder)

    if left_hip is not None and right_hip is not None:
        hip_center = (left_hip + right_hip) * 0.5
        tentative_up = shoulder_center - hip_center
    else:
        tentative_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    if norm(tentative_up) < 1e-6:
        tentative_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    tentative_up = tentative_up - float(np.dot(tentative_up, right_axis)) * right_axis
    if norm(tentative_up) < 1e-6:
        tentative_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up_axis = normalize_vector(tentative_up)

    forward_candidate = np.cross(right_axis, up_axis)
    if norm(forward_candidate) < 1e-6:
        forward_candidate = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_axis = normalize_vector(forward_candidate)
    up_axis = normalize_vector(np.cross(forward_axis, right_axis))

    upper_arm = elbow - shoulder
    forearm = wrist - elbow
    if norm(upper_arm) < 1e-4 or norm(forearm) < 1e-4:
        return None

    upper_arm_body = project_body_frame(upper_arm, right_axis, up_axis, forward_axis)
    forearm_body = project_body_frame(forearm, right_axis, up_axis, forward_axis)

    shoulder_pitch_deg = body_pitch_deg(upper_arm_body)
    shoulder_yaw_deg = -guarded_coronal_plane_deviation_deg(
        upper_arm_body,
        zero_snap_deg=joint_b_zero_snap_deg,
    )
    elbow_yaw_deg = guarded_relative_yaw_deg(
        upper_arm_body,
        forearm_body,
        min_horizontal_ratio=yaw_singularity_ratio,
        zero_snap_deg=joint_c_zero_snap_deg,
    )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Follow your arm with a RealSense D455 and map it to joint_a/joint_b/joint_c.",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="left",
        help="Which arm to follow. Default: left.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORTS["A"],
        help=f"USB-to-CAN COM port. Default: {DEFAULT_PORTS['A']}",
    )
    parser.add_argument(
        "--arm",
        action="store_true",
        help="Arm immediately at startup. Without this flag, preview first and press `a` after calibration to arm.",
    )
    parser.add_argument(
        "--serial-number",
        default=None,
        help="Optional RealSense serial number if more than one device is connected.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the preview window for easier self-view. Tracking still uses the original frame.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="Preview window width.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=720,
        help="Preview window height.",
    )
    parser.add_argument(
        "--color-width",
        type=int,
        default=848,
        help="RealSense color/depth stream width.",
    )
    parser.add_argument(
        "--color-height",
        type=int,
        default=480,
        help="RealSense color/depth stream height.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="RealSense stream FPS.",
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
        "--depth-radius-px",
        type=int,
        default=4,
        help="Patch radius used to sample a stable depth value around each landmark.",
    )
    parser.add_argument(
        "--depth-min-m",
        type=float,
        default=0.20,
        help="Ignore depth samples closer than this distance in meters.",
    )
    parser.add_argument(
        "--depth-max-m",
        type=float,
        default=2.50,
        help="Ignore depth samples farther than this distance in meters.",
    )
    parser.add_argument(
        "--yaw-singularity-ratio",
        type=float,
        default=0.22,
        help="If the arm segment's horizontal projection ratio is below this value, treat yaw as unreliable and snap it to zero.",
    )
    parser.add_argument(
        "--joint-b-zero-snap-deg",
        type=float,
        default=8.0,
        help="Snap shoulder yaw to zero when the measured angle is within this many degrees.",
    )
    parser.add_argument(
        "--joint-c-zero-snap-deg",
        type=float,
        default=10.0,
        help="Snap elbow yaw to zero when the measured angle is within this many degrees.",
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
        help="Temporal filter for joint targets.",
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
    parser.add_argument(
        "--return-zero-on-exit",
        action="store_true",
        help="When quitting with an armed robot, slowly ramp joint_a/joint_b/joint_c back to zero before disconnecting.",
    )
    parser.add_argument(
        "--return-zero-seconds",
        type=float,
        default=2.5,
        help="Ramp duration used with --return-zero-on-exit.",
    )
    parser.add_argument("--joint-a-min", type=float, default=None, help="Override the minimum allowed angle for joint_a.")
    parser.add_argument("--joint-a-max", type=float, default=None, help="Override the maximum allowed angle for joint_a.")
    parser.add_argument("--joint-b-min", type=float, default=None, help="Override the minimum allowed angle for joint_b.")
    parser.add_argument("--joint-b-max", type=float, default=None, help="Override the maximum allowed angle for joint_b.")
    parser.add_argument("--joint-c-min", type=float, default=None, help="Override the minimum allowed angle for joint_c.")
    parser.add_argument("--joint-c-max", type=float, default=None, help="Override the maximum allowed angle for joint_c.")
    return resolve_side_defaults(parser.parse_args())


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


def apply_joint_targets(controller, targets: JointTargets) -> None:
    controller.set_target("joint_a", targets.joint_a)
    controller.set_target("joint_b", targets.joint_b)
    controller.set_target("joint_c", targets.joint_c)
    controller.command_motor("joint_a", force=True)
    controller.command_motor("joint_b", force=True)
    controller.command_motor("joint_c", force=True)


def arm_robot(port: str, motor_specs) -> tuple:
    controller = initialize_controller(port, motor_specs)
    home_targets = current_robot_home(controller)
    print(
        "[robot] armed on "
        f"{port} with home joint targets "
        f"A={home_targets.joint_a:.1f}, B={home_targets.joint_b:.1f}, C={home_targets.joint_c:.1f}"
    )
    return controller, home_targets


def ramp_robot_to_zero(controller, duration_s: float, step_s: float = 0.05) -> None:
    duration_s = max(0.1, duration_s)
    steps = max(2, int(round(duration_s / max(0.01, step_s))))
    start = current_robot_home(controller)
    end = JointTargets(0.0, 0.0, 0.0)
    print(f"[exit] ramping joints to zero over {duration_s:.1f}s")
    for step_index in range(1, steps + 1):
        alpha = step_index / steps
        targets = JointTargets(
            joint_a=start.joint_a + (end.joint_a - start.joint_a) * alpha,
            joint_b=start.joint_b + (end.joint_b - start.joint_b) * alpha,
            joint_c=start.joint_c + (end.joint_c - start.joint_c) * alpha,
        )
        apply_joint_targets(controller, targets)
        controller.refresh_states()
        controller.maintain_target_holds()
        time.sleep(duration_s / steps)
    apply_joint_targets(controller, end)
    time.sleep(0.15)


def update_calibration(
    calibration: Calibration | None,
    raw_pose: RawArmPose,
    *,
    update_pitch: bool,
    update_shoulder_yaw: bool,
    update_elbow_yaw: bool,
) -> Calibration:
    if calibration is None:
        calibration = Calibration(
            shoulder_pitch_deg=raw_pose.shoulder_pitch_deg,
            shoulder_yaw_deg=raw_pose.shoulder_yaw_deg,
            elbow_yaw_deg=raw_pose.elbow_yaw_deg,
        )
    return replace(
        calibration,
        shoulder_pitch_deg=raw_pose.shoulder_pitch_deg if update_pitch else calibration.shoulder_pitch_deg,
        shoulder_yaw_deg=raw_pose.shoulder_yaw_deg if update_shoulder_yaw else calibration.shoulder_yaw_deg,
        elbow_yaw_deg=raw_pose.elbow_yaw_deg if update_elbow_yaw else calibration.elbow_yaw_deg,
    )


def main() -> int:
    args = parse_args()
    motor_specs = build_motor_specs(args)
    limits_by_key = {spec.key: spec for spec in motor_specs}
    tracker: RealSenseTracker | None = None
    controller = None
    window_name = "Robot Arm RealSense D455 Control"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, max(320, args.window_width), max(240, args.window_height))

    try:
        tracker = RealSenseTracker(
            color_width=args.color_width,
            color_height=args.color_height,
            fps=args.fps,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            serial_number=args.serial_number,
        )
    except Exception as exc:
        print(f"[realsense] {exc}", file=sys.stderr)
        return 1

    try:
        if args.arm:
            controller, home_targets = arm_robot(args.port, motor_specs)
            follow_enabled = True
        else:
            home_targets = JointTargets(0.0, 0.0, 0.0)
            follow_enabled = False
            print("[robot] preview only. Add --arm to actually move the robot.")

        print(
            "[limits] "
            f"joint_a=[{limits_by_key['joint_a'].min_deg:.1f}, {limits_by_key['joint_a'].max_deg:.1f}] "
            f"joint_b=[{limits_by_key['joint_b'].min_deg:.1f}, {limits_by_key['joint_b'].max_deg:.1f}] "
            f"joint_c=[{limits_by_key['joint_c'].min_deg:.1f}, {limits_by_key['joint_c'].max_deg:.1f}]"
        )
        print(
            "[depth] "
            f"stream={args.color_width}x{args.color_height}@{args.fps} "
            f"range={args.depth_min_m:.2f}-{args.depth_max_m:.2f}m "
            f"patch_radius={args.depth_radius_px}px"
        )
        print(
            "[yaw-guard] "
            f"singularity_ratio={args.yaw_singularity_ratio:.2f} "
            f"joint_b_zero_snap={args.joint_b_zero_snap_deg:.1f}deg "
            f"joint_c_zero_snap={args.joint_c_zero_snap_deg:.1f}deg"
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
        print(f"[safety] temp_limit={DEFAULT_TEMP_LIMIT_C:.1f}C")
        print("[input] press p for joint_a, b for joint_b, c for joint_c, n for full neutral, a to arm/toggle follow, q to quit.")

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
            ok, frame, depth_frame = tracker.read()
            if not ok or frame is None or depth_frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "RealSense read failed",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, blank)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tracker.pose.process(rgb)
            raw_pose = extract_depth_arm_pose(
                results=results,
                depth_frame=depth_frame,
                side=args.side,
                min_visibility=args.min_visibility,
                depth_radius_px=args.depth_radius_px,
                depth_min_m=args.depth_min_m,
                depth_max_m=args.depth_max_m,
                yaw_singularity_ratio=args.yaw_singularity_ratio,
                joint_b_zero_snap_deg=args.joint_b_zero_snap_deg,
                joint_c_zero_snap_deg=args.joint_c_zero_snap_deg,
            )

            if raw_pose is not None and calibration is None:
                calibration = Calibration(
                    shoulder_pitch_deg=raw_pose.shoulder_pitch_deg,
                    shoulder_yaw_deg=raw_pose.shoulder_yaw_deg,
                    elbow_yaw_deg=raw_pose.elbow_yaw_deg,
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

            joint_targets = None
            if raw_pose is not None and calibration is not None:
                raw_targets = raw_to_joint_targets(
                    raw_pose=raw_pose,
                    calibration=calibration,
                    home_targets=home_targets,
                    limits_by_key=limits_by_key,
                    args=args,
                )
                filtered_targets = smoother.update(raw_targets)
                if temporal_filter is not None:
                    if isinstance(temporal_filter, OneEuroJointFilter):
                        filtered_targets = temporal_filter.update(filtered_targets, now_ts=time.time())
                    else:
                        filtered_targets = temporal_filter.update(filtered_targets)

                control_targets = filtered_targets
                if pid_controller is not None:
                    control_targets = pid_controller.update(
                        filtered_targets,
                        measured=current_joint_measurement(controller),
                        now_ts=time.time(),
                    )
                if rate_limiter is not None:
                    control_targets = rate_limiter.update(control_targets, now_ts=time.time())
                joint_targets = control_targets

            print_joint_targets_live(
                active_camera_index=0,
                raw_pose=raw_pose,
                joint_targets=joint_targets,
                arm_enabled=controller is not None and follow_enabled,
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

                if (
                    follow_enabled
                    and joint_targets is not None
                    and (send_interval_s <= 0.0 or now - last_sent_at >= send_interval_s)
                ):
                    apply_joint_targets(controller, joint_targets)
                    last_sent_at = now

            display_frame = frame.copy()
            if args.mirror:
                display_frame = cv2.flip(display_frame, 1)
            display_frame = draw_arm_skeleton(
                frame=display_frame,
                results=results,
                side=args.side,
                min_visibility=args.min_visibility,
                mirror=args.mirror,
                active=raw_pose is not None,
            )
            status_text = "depth locked" if raw_pose is not None else "depth pose not locked"
            display_frame = annotate_frame(
                frame=display_frame,
                camera_index=0,
                active=raw_pose is not None,
                raw_pose=raw_pose,
                joint_targets=joint_targets,
                status_text=status_text,
            )
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("a"):
                if controller is None:
                    if calibration is None:
                        print("[robot] calibrate first, then press a to arm.")
                    else:
                        controller, home_targets = arm_robot(args.port, motor_specs)
                        follow_enabled = True
                        if pid_controller is not None:
                            pid_controller.reset(output=current_joint_measurement(controller) or home_targets)
                else:
                    follow_enabled = not follow_enabled
                    if follow_enabled:
                        print("[robot] follow enabled")
                        if pid_controller is not None:
                            pid_controller.reset(output=current_joint_measurement(controller) or home_targets)
                    else:
                        print("[robot] follow paused")
                        try:
                            controller.hold_all()
                        except Exception:
                            pass
            if key == ord("n") and raw_pose is not None:
                calibration = update_calibration(
                    calibration,
                    raw_pose,
                    update_pitch=True,
                    update_shoulder_yaw=True,
                    update_elbow_yaw=True,
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
            if key == ord("p") and raw_pose is not None:
                calibration = update_calibration(
                    calibration,
                    raw_pose,
                    update_pitch=True,
                    update_shoulder_yaw=False,
                    update_elbow_yaw=False,
                )
                if pid_controller is not None:
                    pid_controller.reset(output=current_joint_measurement(controller) or home_targets)
                print()
                print(f"[calibration] updated joint_a baseline to pitch={calibration.shoulder_pitch_deg:+.1f}")
            if key == ord("b") and raw_pose is not None:
                calibration = update_calibration(
                    calibration,
                    raw_pose,
                    update_pitch=False,
                    update_shoulder_yaw=True,
                    update_elbow_yaw=False,
                )
                if pid_controller is not None:
                    pid_controller.reset(output=current_joint_measurement(controller) or home_targets)
                print()
                print(f"[calibration] updated joint_b baseline to shoulder_yaw={calibration.shoulder_yaw_deg:+.1f}")
            if key == ord("c") and raw_pose is not None:
                calibration = update_calibration(
                    calibration,
                    raw_pose,
                    update_pitch=False,
                    update_shoulder_yaw=False,
                    update_elbow_yaw=True,
                )
                if pid_controller is not None:
                    pid_controller.reset(output=current_joint_measurement(controller) or home_targets)
                print()
                print(f"[calibration] updated joint_c baseline to elbow_yaw={calibration.elbow_yaw_deg:+.1f}")

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
                if args.return_zero_on_exit:
                    ramp_robot_to_zero(controller, duration_s=args.return_zero_seconds)
                else:
                    controller.hold_all()
            except Exception:
                pass
            controller.disconnect_all()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
