from __future__ import annotations

"""3-axis IK demo for a shoulder-pitch + yaw + yaw robot arm.

This version matches the clarified mechanism more closely:
- joint_a: shoulder pitch, lifting the whole arm
- joint_b: yaw
- joint_c: yaw

The app shows a top view (X/Y) and a side view (X/Z), solves a simplified
analytic IK for target position, and writes a small demo URDF that can be used
to validate forward kinematics with ikpy when it is available.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    from PyQt6.QtCore import QPointF, Qt, pyqtSignal
    from PyQt6.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QPlainTextEdit,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit("PyQt6 is required. Install it with `pip install PyQt6`.") from exc

try:
    from ikpy.chain import Chain

    IKPY_AVAILABLE = True
except ImportError:
    Chain = None
    IKPY_AVAILABLE = False


WINDOW_TITLE = "3-Axis Mixed-Axis IK Demo"
MM_SUFFIX = " mm"
DEG_SUFFIX = " deg"
TARGET_MARGIN_MM = 40.0
GENERATED_URDF_PATH = Path(__file__).resolve().parent / "assets" / "generated_mixed_axis_arm.urdf"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def normalize_deg(value: float) -> float:
    wrapped = (value + 180.0) % 360.0 - 180.0
    if wrapped == -180.0 and value > 0.0:
        return 180.0
    return wrapped


def angular_error_deg(a: float, b: float) -> float:
    return abs(normalize_deg(a - b))


@dataclass(frozen=True)
class JointLimit:
    minimum_deg: float
    maximum_deg: float

    def contains(self, value: float, tolerance_deg: float = 1e-6) -> bool:
        return self.minimum_deg - tolerance_deg <= value <= self.maximum_deg + tolerance_deg

    def violation_deg(self, value: float) -> float:
        if value < self.minimum_deg:
            return self.minimum_deg - value
        if value > self.maximum_deg:
            return value - self.maximum_deg
        return 0.0


@dataclass(frozen=True)
class ArmConfig:
    link_lengths_mm: tuple[float, float, float]
    yaw_plane_height_mm: float
    joint_limits: tuple[JointLimit, JointLimit, JointLimit]
    gear_ratios: tuple[float, float, float]

    @property
    def reach_radius_mm(self) -> float:
        return sum(self.link_lengths_mm) + abs(self.yaw_plane_height_mm)


@dataclass
class IKCandidate:
    angles_deg: tuple[float, float, float]
    frames_xyz: list[tuple[float, float, float]]
    within_limits: bool
    limit_violation_deg: float
    movement_score: float
    branch_name: str


@dataclass
class IKResult:
    success: bool
    angles_deg: tuple[float, float, float] | None
    frames_xyz: list[tuple[float, float, float]]
    target_xyz: tuple[float, float, float]
    solver_name: str
    message: str
    within_limits: bool = False
    position_error_mm: float = 0.0
    urdf_tip_error_mm: float | None = None

    @property
    def points_xyz(self) -> list[tuple[float, float, float]]:
        if len(self.frames_xyz) >= 5:
            return [self.frames_xyz[0], self.frames_xyz[2], self.frames_xyz[3], self.frames_xyz[4]]
        return self.frames_xyz


class MixedAxisArmKinematics:
    """Simplified kinematics for shoulder pitch + yaw + yaw."""

    def __init__(self, config: ArmConfig):
        self.config = config

    def joint_to_motor_angles(self, angles_deg: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(
            angle_deg * ratio for angle_deg, ratio in zip(angles_deg, self.config.gear_ratios)
        )

    def forward_frames_xyz(self, angles_deg: tuple[float, float, float]) -> list[tuple[float, float, float]]:
        shoulder_deg, yaw_b_deg, yaw_c_deg = angles_deg
        link_1, link_2, link_3 = self.config.link_lengths_mm
        height_mm = self.config.yaw_plane_height_mm

        yaw_b_rad = math.radians(yaw_b_deg)
        yaw_c_rad = math.radians(yaw_c_deg)

        base_frame = (0.0, 0.0, 0.0)
        shoulder_origin = (0.0, 0.0, 0.0)
        joint_b_local = (link_1, 0.0, height_mm)
        joint_c_local = (
            link_1 + link_2 * math.cos(yaw_b_rad),
            link_2 * math.sin(yaw_b_rad),
            height_mm,
        )
        tip_local = (
            joint_c_local[0] + link_3 * math.cos(yaw_b_rad + yaw_c_rad),
            joint_c_local[1] + link_3 * math.sin(yaw_b_rad + yaw_c_rad),
            height_mm,
        )

        return [
            base_frame,
            shoulder_origin,
            self._rotate_about_y(joint_b_local, shoulder_deg),
            self._rotate_about_y(joint_c_local, shoulder_deg),
            self._rotate_about_y(tip_local, shoulder_deg),
        ]

    def solve_analytic(
        self,
        target_xyz: tuple[float, float, float],
        current_angles_deg: tuple[float, float, float],
        branch_mode: Literal["auto", "positive", "negative"] = "auto",
    ) -> IKResult:
        target_x_mm, target_y_mm, target_z_mm = target_xyz
        link_1, link_2, link_3 = self.config.link_lengths_mm
        height_mm = self.config.yaw_plane_height_mm

        radial_xz_mm = math.hypot(target_x_mm, target_z_mm)
        if radial_xz_mm + 1e-7 < abs(height_mm):
            return IKResult(
                success=False,
                angles_deg=None,
                frames_xyz=self.forward_frames_xyz(current_angles_deg),
                target_xyz=target_xyz,
                solver_name="analytic-3d",
                message="Target is too close to the shoulder axis for the configured height offset.",
            )

        local_target_x = math.sqrt(max(0.0, radial_xz_mm**2 - height_mm**2))
        shoulder_deg = normalize_deg(
            math.degrees(math.atan2(height_mm, local_target_x) - math.atan2(target_z_mm, target_x_mm))
        )
        planar_target_x = local_target_x - link_1
        planar_target_y = target_y_mm

        numerator = planar_target_x**2 + planar_target_y**2 - link_2**2 - link_3**2
        denominator = 2.0 * link_2 * link_3
        cos_joint_c = numerator / denominator if denominator else 0.0

        if abs(cos_joint_c) > 1.0 + 1e-7:
            return IKResult(
                success=False,
                angles_deg=None,
                frames_xyz=self.forward_frames_xyz(current_angles_deg),
                target_xyz=target_xyz,
                solver_name="analytic-3d",
                message="Target is outside the mixed-axis workspace for the configured link lengths.",
            )

        cos_joint_c = clamp(cos_joint_c, -1.0, 1.0)
        sin_joint_c = math.sqrt(max(0.0, 1.0 - cos_joint_c**2))
        branch_signs = {"positive": (1.0,), "negative": (-1.0,), "auto": (1.0, -1.0)}[branch_mode]

        candidates: list[IKCandidate] = []
        for branch_sign in branch_signs:
            joint_c_rad = math.atan2(branch_sign * sin_joint_c, cos_joint_c)
            joint_b_rad = math.atan2(planar_target_y, planar_target_x) - math.atan2(
                link_3 * math.sin(joint_c_rad),
                link_2 + link_3 * math.cos(joint_c_rad),
            )

            angles_deg = (
                shoulder_deg,
                normalize_deg(math.degrees(joint_b_rad)),
                normalize_deg(math.degrees(joint_c_rad)),
            )
            frames_xyz = self.forward_frames_xyz(angles_deg)
            limit_violation_deg = sum(
                limit.violation_deg(angle_deg)
                for limit, angle_deg in zip(self.config.joint_limits, angles_deg)
            )
            movement_score = sum(
                angular_error_deg(candidate_angle, current_angle)
                for candidate_angle, current_angle in zip(angles_deg, current_angles_deg)
            )
            candidates.append(
                IKCandidate(
                    angles_deg=angles_deg,
                    frames_xyz=frames_xyz,
                    within_limits=limit_violation_deg <= 1e-6,
                    limit_violation_deg=limit_violation_deg,
                    movement_score=movement_score,
                    branch_name="positive-yaw" if branch_sign > 0.0 else "negative-yaw",
                )
            )

        valid_candidates = [candidate for candidate in candidates if candidate.within_limits]
        if valid_candidates:
            best = min(valid_candidates, key=lambda candidate: candidate.movement_score)
            return IKResult(
                success=True,
                angles_deg=best.angles_deg,
                frames_xyz=best.frames_xyz,
                target_xyz=target_xyz,
                solver_name="analytic-3d",
                message=f"Analytic IK solved using the {best.branch_name} branch.",
                within_limits=True,
                position_error_mm=self._position_error_mm(best.frames_xyz[-1], target_xyz),
            )

        best = min(candidates, key=lambda candidate: (candidate.limit_violation_deg, candidate.movement_score))
        return IKResult(
            success=False,
            angles_deg=best.angles_deg,
            frames_xyz=best.frames_xyz,
            target_xyz=target_xyz,
            solver_name="analytic-3d",
            message=(
                "The target is geometrically solvable, but at least one joint would exceed the current limits."
            ),
            within_limits=False,
            position_error_mm=self._position_error_mm(best.frames_xyz[-1], target_xyz),
        )

    def write_demo_urdf(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.build_urdf_text(), encoding="utf-8")

    def build_urdf_text(self) -> str:
        link_1, link_2, link_3 = self.config.link_lengths_mm
        height_mm = self.config.yaw_plane_height_mm
        limit_a, limit_b, limit_c = self.config.joint_limits
        return f"""<?xml version="1.0"?>
<robot name="mixed_axis_arm">
  <link name="base_link"/>

  <joint name="joint_a" type="revolute">
    <parent link="base_link"/>
    <child link="link_a"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="{math.radians(limit_a.minimum_deg):.10f}" upper="{math.radians(limit_a.maximum_deg):.10f}" effort="1" velocity="1"/>
  </joint>
  <link name="link_a"/>

  <joint name="joint_b" type="revolute">
    <parent link="link_a"/>
    <child link="link_b"/>
    <origin xyz="{link_1:.6f} 0 {height_mm:.6f}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="{math.radians(limit_b.minimum_deg):.10f}" upper="{math.radians(limit_b.maximum_deg):.10f}" effort="1" velocity="1"/>
  </joint>
  <link name="link_b"/>

  <joint name="joint_c" type="revolute">
    <parent link="link_b"/>
    <child link="link_c"/>
    <origin xyz="{link_2:.6f} 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="{math.radians(limit_c.minimum_deg):.10f}" upper="{math.radians(limit_c.maximum_deg):.10f}" effort="1" velocity="1"/>
  </joint>
  <link name="link_c"/>

  <joint name="tool_fixed" type="fixed">
    <parent link="link_c"/>
    <child link="tool_link"/>
    <origin xyz="{link_3:.6f} 0 0" rpy="0 0 0"/>
  </joint>
  <link name="tool_link"/>
</robot>
"""

    @staticmethod
    def _rotate_about_y(local_xyz: tuple[float, float, float], shoulder_deg: float) -> tuple[float, float, float]:
        x_mm, y_mm, z_mm = local_xyz
        shoulder_rad = math.radians(shoulder_deg)
        cos_theta = math.cos(shoulder_rad)
        sin_theta = math.sin(shoulder_rad)
        world_x = cos_theta * x_mm + sin_theta * z_mm
        world_y = y_mm
        world_z = -sin_theta * x_mm + cos_theta * z_mm
        return (world_x, world_y, world_z)

    @staticmethod
    def _position_error_mm(actual_xyz: tuple[float, float, float], target_xyz: tuple[float, float, float]) -> float:
        dx = actual_xyz[0] - target_xyz[0]
        dy = actual_xyz[1] - target_xyz[1]
        dz = actual_xyz[2] - target_xyz[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)


class URDFValidator:
    def __init__(self, urdf_path: Path):
        if not IKPY_AVAILABLE:
            raise RuntimeError("ikpy is not installed")
        self.urdf_path = urdf_path
        self.chain = Chain.from_urdf_file(
            str(urdf_path),
            base_elements=["base_link"],
            active_links_mask=[False, True, True, True, False],
            symbolic=False,
            name="mixed_axis_arm",
        )

    def forward_frames_xyz(self, angles_deg: tuple[float, float, float]) -> list[tuple[float, float, float]]:
        joint_values_rad = [0.0] + [math.radians(value) for value in angles_deg] + [0.0]
        frames = self.chain.forward_kinematics(joint_values_rad, full_kinematics=True)
        return [
            tuple(float(value) for value in frame[:3, 3])
            for frame in frames
        ]


class ProjectionCanvas(QWidget):
    targetChanged = pyqtSignal(float, float)

    def __init__(self, title: str, horizontal_axis: str, vertical_axis: str):
        super().__init__()
        self.title = title
        self.horizontal_axis = horizontal_axis
        self.vertical_axis = vertical_axis
        self.config = ArmConfig(
            link_lengths_mm=(280.0, 235.0, 150.0),
            yaw_plane_height_mm=40.0,
            joint_limits=(
                JointLimit(-60.0, 0.0),
                JointLimit(-75.0, 75.0),
                JointLimit(-90.0, 90.0),
            ),
            gear_ratios=(2.0, 2.0, 2.0),
        )
        self.points_xyz = [(0.0, 0.0, 0.0)]
        self.target_xyz = (0.0, 0.0, 0.0)
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_scene(
        self,
        config: ArmConfig,
        points_xyz: list[tuple[float, float, float]],
        target_xyz: tuple[float, float, float],
    ) -> None:
        self.config = config
        self.points_xyz = points_xyz
        self.target_xyz = target_xyz
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            return
        horizontal_value, vertical_value = self._screen_to_plane(event.position().x(), event.position().y())
        self.targetChanged.emit(horizontal_value, vertical_value)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#f8fafc"))
        self._draw_grid(painter)
        self._draw_axes(painter)
        self._draw_target(painter)
        self._draw_arm(painter)
        self._draw_title(painter)

    def _draw_grid(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        span_mm = self._world_span_mm()
        grid_step_mm = 100.0
        tick_count = int(span_mm / grid_step_mm)
        for tick in range(-tick_count, tick_count + 1):
            world_value = tick * grid_step_mm
            x1, y1 = self._plane_to_screen(world_value, -span_mm)
            x2, y2 = self._plane_to_screen(world_value, span_mm)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            x3, y3 = self._plane_to_screen(-span_mm, world_value)
            x4, y4 = self._plane_to_screen(span_mm, world_value)
            painter.drawLine(QPointF(x3, y3), QPointF(x4, y4))

    def _draw_axes(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#94a3b8"), 1.5))
        left_x, center_y = self._plane_to_screen(-self._world_span_mm(), 0.0)
        right_x, _ = self._plane_to_screen(self._world_span_mm(), 0.0)
        center_x, top_y = self._plane_to_screen(0.0, self._world_span_mm())
        _, bottom_y = self._plane_to_screen(0.0, -self._world_span_mm())
        painter.drawLine(QPointF(left_x, center_y), QPointF(right_x, center_y))
        painter.drawLine(QPointF(center_x, top_y), QPointF(center_x, bottom_y))

    def _draw_target(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#ef4444"), 2.0))
        x_px, y_px = self._project_to_screen(self.target_xyz)
        cross_half = 7.0
        painter.drawLine(QPointF(x_px - cross_half, y_px - cross_half), QPointF(x_px + cross_half, y_px + cross_half))
        painter.drawLine(QPointF(x_px - cross_half, y_px + cross_half), QPointF(x_px + cross_half, y_px - cross_half))

    def _draw_arm(self, painter: QPainter) -> None:
        if len(self.points_xyz) < 2:
            return
        link_colors = ("#2563eb", "#0f766e", "#ca8a04")
        for index in range(len(self.points_xyz) - 1):
            start_px = self._project_to_screen(self.points_xyz[index])
            end_px = self._project_to_screen(self.points_xyz[index + 1])
            painter.setPen(QPen(QColor(link_colors[index]), 8.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(*start_px), QPointF(*end_px))

        painter.setPen(QPen(QColor("#0f172a"), 2.0))
        painter.setBrush(QColor("#ffffff"))
        for point_xyz in self.points_xyz:
            point_px = self._project_to_screen(point_xyz)
            painter.drawEllipse(QPointF(*point_px), 6.5, 6.5)

    def _draw_title(self, painter: QPainter) -> None:
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        painter.drawText(14, 26, self.title)
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(14, 44, f"Click to edit {self.horizontal_axis.upper()} / {self.vertical_axis.upper()} target")

    def _project_to_screen(self, point_xyz: tuple[float, float, float]) -> tuple[float, float]:
        horizontal_value, vertical_value = self._project(point_xyz)
        return self._plane_to_screen(horizontal_value, vertical_value)

    def _project(self, point_xyz: tuple[float, float, float]) -> tuple[float, float]:
        axis_index = {"x": 0, "y": 1, "z": 2}
        return (
            point_xyz[axis_index[self.horizontal_axis]],
            point_xyz[axis_index[self.vertical_axis]],
        )

    def _world_span_mm(self) -> float:
        return self.config.reach_radius_mm + TARGET_MARGIN_MM

    def _scale_px_per_mm(self) -> float:
        usable = min(self.width(), self.height()) - 44.0
        return usable / (2.0 * self._world_span_mm())

    def _plane_to_screen(self, horizontal_value: float, vertical_value: float) -> tuple[float, float]:
        center_x = self.width() / 2.0
        center_y = self.height() / 2.0
        scale = self._scale_px_per_mm()
        return (
            center_x + horizontal_value * scale,
            center_y - vertical_value * scale,
        )

    def _screen_to_plane(self, x_px: float, y_px: float) -> tuple[float, float]:
        center_x = self.width() / 2.0
        center_y = self.height() / 2.0
        scale = self._scale_px_per_mm()
        return (
            (x_px - center_x) / scale,
            (center_y - y_px) / scale,
        )


class Orbit3DCanvas(QWidget):
    """Simple software-rendered 3D preview with orbit controls."""

    targetChanged3D = pyqtSignal(float, float, float)

    def __init__(self):
        super().__init__()
        self.config = ArmConfig(
            link_lengths_mm=(280.0, 235.0, 150.0),
            yaw_plane_height_mm=40.0,
            joint_limits=(
                JointLimit(-60.0, 0.0),
                JointLimit(-75.0, 75.0),
                JointLimit(-90.0, 90.0),
            ),
            gear_ratios=(2.0, 2.0, 2.0),
        )
        self.points_xyz = [(0.0, 0.0, 0.0)]
        self.target_xyz = (0.0, 0.0, 0.0)
        self.camera_yaw_deg = -35.0
        self.camera_pitch_deg = 22.0
        self.zoom = 1.0
        self._drag_mode: Literal["orbit", "handle"] | None = None
        self._drag_last_xy: tuple[float, float] | None = None
        self._drag_axis: str | None = None
        self._drag_start_target: tuple[float, float, float] | None = None
        self._drag_start_scalar: float = 0.0
        self.setMinimumHeight(360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_scene(
        self,
        config: ArmConfig,
        points_xyz: list[tuple[float, float, float]],
        target_xyz: tuple[float, float, float],
    ) -> None:
        self.config = config
        self.points_xyz = points_xyz
        self.target_xyz = target_xyz
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            mouse_xy = (event.position().x(), event.position().y())
            picked_axis = self._pick_handle(*mouse_xy)
            if picked_axis is not None:
                self._drag_mode = "handle"
                self._drag_axis = picked_axis
                self._drag_start_target = self.target_xyz
                self._drag_start_scalar = self._axis_scalar(
                    picked_axis,
                    mouse_xy[0],
                    mouse_xy[1],
                    self._drag_start_target,
                )
            else:
                self._drag_mode = "orbit"
                self._drag_last_xy = mouse_xy

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        current_xy = (event.position().x(), event.position().y())
        if self._drag_mode == "handle" and self._drag_axis and self._drag_start_target is not None:
            self.target_xyz = self._drag_target_along_axis(
                self._drag_axis,
                current_xy,
                self._drag_start_target,
                self._drag_start_scalar,
            )
            self.targetChanged3D.emit(*self.target_xyz)
            self.update()
            return
        if self._drag_mode == "orbit" and self._drag_last_xy is not None:
            delta_x = current_xy[0] - self._drag_last_xy[0]
            delta_y = current_xy[1] - self._drag_last_xy[1]
            self.camera_yaw_deg += delta_x * 0.45
            self.camera_pitch_deg = clamp(self.camera_pitch_deg + delta_y * 0.30, -80.0, 80.0)
            self._drag_last_xy = current_xy
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = None
            self._drag_last_xy = None
            self._drag_axis = None
            self._drag_start_target = None

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        step = event.angleDelta().y() / 120.0
        self.zoom = clamp(self.zoom * (1.1 ** step), 0.4, 3.0)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#f8fafc"))
        self._draw_floor_grid(painter)
        self._draw_axes(painter)
        self._draw_target(painter)
        self._draw_arm(painter)
        self._draw_title(painter)

    def _draw_floor_grid(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#dbe4ef"), 1))
        span_mm = self._world_span_mm()
        grid_step_mm = 100.0
        tick_count = int(span_mm / grid_step_mm)
        for tick in range(-tick_count, tick_count + 1):
            coord = tick * grid_step_mm
            start_a = self._project_point((coord, -span_mm, 0.0))
            end_a = self._project_point((coord, span_mm, 0.0))
            start_b = self._project_point((-span_mm, coord, 0.0))
            end_b = self._project_point((span_mm, coord, 0.0))
            painter.drawLine(QPointF(start_a[0], start_a[1]), QPointF(end_a[0], end_a[1]))
            painter.drawLine(QPointF(start_b[0], start_b[1]), QPointF(end_b[0], end_b[1]))

    def _draw_axes(self, painter: QPainter) -> None:
        axes = (
            ((0.0, 0.0, 0.0), (220.0, 0.0, 0.0), "#2563eb", "X"),
            ((0.0, 0.0, 0.0), (0.0, 220.0, 0.0), "#0f766e", "Y"),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 220.0), "#ca8a04", "Z"),
        )
        for start_xyz, end_xyz, color, label in axes:
            start_px = self._project_point(start_xyz)
            end_px = self._project_point(end_xyz)
            painter.setPen(QPen(QColor(color), 2.0))
            painter.drawLine(QPointF(start_px[0], start_px[1]), QPointF(end_px[0], end_px[1]))
            painter.setPen(QColor(color))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            painter.drawText(int(end_px[0] + 6), int(end_px[1] - 4), label)

    def _draw_target(self, painter: QPainter) -> None:
        x_px, y_px, _ = self._project_point(self.target_xyz)
        painter.setPen(QPen(QColor("#ef4444"), 2.0))
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QPointF(x_px, y_px), 7.0, 7.0)
        cross_half = 8.0
        painter.drawLine(QPointF(x_px - cross_half, y_px), QPointF(x_px + cross_half, y_px))
        painter.drawLine(QPointF(x_px, y_px - cross_half), QPointF(x_px, y_px + cross_half))
        self._draw_handles(painter)

    def _draw_arm(self, painter: QPainter) -> None:
        if len(self.points_xyz) < 2:
            return

        link_colors = ("#2563eb", "#0f766e", "#ca8a04")
        segments: list[tuple[float, tuple[float, float], tuple[float, float], str]] = []
        for index in range(len(self.points_xyz) - 1):
            start_proj = self._project_point(self.points_xyz[index])
            end_proj = self._project_point(self.points_xyz[index + 1])
            avg_depth = (start_proj[2] + end_proj[2]) / 2.0
            segments.append((avg_depth, (start_proj[0], start_proj[1]), (end_proj[0], end_proj[1]), link_colors[index]))

        for _, start_px, end_px, color in sorted(segments, key=lambda item: item[0]):
            painter.setPen(QPen(QColor(color), 8.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(*start_px), QPointF(*end_px))

        projected_points = [self._project_point(point_xyz) for point_xyz in self.points_xyz]
        for _, x_px, y_px in sorted((depth, x_px, y_px) for x_px, y_px, depth in projected_points):
            painter.setPen(QPen(QColor("#0f172a"), 2.0))
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QPointF(x_px, y_px), 6.5, 6.5)

    def _draw_title(self, painter: QPainter) -> None:
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        painter.drawText(14, 26, "3D View")
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(14, 44, "Drag empty space to orbit, drag X/Y/Z handles to move target, wheel to zoom")

    def _draw_handles(self, painter: QPainter) -> None:
        axis_styles = (
            ("x", "#2563eb"),
            ("y", "#0f766e"),
            ("z", "#ca8a04"),
        )
        for axis_name, color in axis_styles:
            start_px, end_px = self._handle_segment_screen(axis_name, self.target_xyz)
            width = 4.0 if self._drag_axis == axis_name else 3.0
            painter.setPen(QPen(QColor(color), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(*start_px), QPointF(*end_px))
            painter.setBrush(QColor(color))
            painter.drawEllipse(QPointF(*end_px), 5.5, 5.5)

    def _pick_handle(self, x_px: float, y_px: float) -> str | None:
        best_axis: str | None = None
        best_distance = 14.0
        for axis_name in ("x", "y", "z"):
            start_px, end_px = self._handle_segment_screen(axis_name, self.target_xyz)
            distance = self._distance_to_segment((x_px, y_px), start_px, end_px)
            if distance < best_distance:
                best_distance = distance
                best_axis = axis_name
        return best_axis

    def _handle_segment_screen(
        self,
        axis_name: str,
        target_xyz: tuple[float, float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        start_proj = self._project_point(target_xyz)
        axis_length_mm = self._handle_axis_length_mm()
        target_as_list = list(target_xyz)
        axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
        target_as_list[axis_index] += axis_length_mm
        end_proj = self._project_point(tuple(target_as_list))
        return ((start_proj[0], start_proj[1]), (end_proj[0], end_proj[1]))

    def _axis_scalar(
        self,
        axis_name: str,
        x_px: float,
        y_px: float,
        target_xyz: tuple[float, float, float],
    ) -> float:
        start_px, end_px = self._handle_segment_screen(axis_name, target_xyz)
        dir_x = end_px[0] - start_px[0]
        dir_y = end_px[1] - start_px[1]
        length_sq = dir_x * dir_x + dir_y * dir_y
        if length_sq <= 1e-9:
            return 0.0
        return ((x_px - start_px[0]) * dir_x + (y_px - start_px[1]) * dir_y) / length_sq

    def _drag_target_along_axis(
        self,
        axis_name: str,
        mouse_xy: tuple[float, float],
        start_target: tuple[float, float, float],
        start_scalar: float,
    ) -> tuple[float, float, float]:
        current_scalar = self._axis_scalar(axis_name, mouse_xy[0], mouse_xy[1], start_target)
        delta_mm = (current_scalar - start_scalar) * self._handle_axis_length_mm()
        axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
        target_values = list(start_target)
        target_range = self._world_span_mm()
        target_values[axis_index] = clamp(target_values[axis_index] + delta_mm, -target_range, target_range)
        return tuple(target_values)

    def _handle_axis_length_mm(self) -> float:
        return min(140.0, self.config.reach_radius_mm * 0.22)

    @staticmethod
    def _distance_to_segment(
        point_xy: tuple[float, float],
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
    ) -> float:
        dir_x = end_xy[0] - start_xy[0]
        dir_y = end_xy[1] - start_xy[1]
        length_sq = dir_x * dir_x + dir_y * dir_y
        if length_sq <= 1e-9:
            return math.dist(point_xy, start_xy)
        t_value = ((point_xy[0] - start_xy[0]) * dir_x + (point_xy[1] - start_xy[1]) * dir_y) / length_sq
        t_value = clamp(t_value, 0.0, 1.0)
        closest_xy = (start_xy[0] + dir_x * t_value, start_xy[1] + dir_y * t_value)
        return math.dist(point_xy, closest_xy)

    def _project_point(self, point_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        x_mm, y_mm, z_mm = point_xyz
        center_x_mm = self.config.reach_radius_mm * 0.45
        translated_x = x_mm - center_x_mm
        translated_y = y_mm
        translated_z = z_mm

        yaw_rad = math.radians(self.camera_yaw_deg)
        pitch_rad = math.radians(self.camera_pitch_deg)

        x_after_yaw = math.cos(yaw_rad) * translated_x - math.sin(yaw_rad) * translated_y
        y_after_yaw = math.sin(yaw_rad) * translated_x + math.cos(yaw_rad) * translated_y
        z_after_yaw = translated_z

        x_camera = x_after_yaw
        y_camera = math.cos(pitch_rad) * y_after_yaw - math.sin(pitch_rad) * z_after_yaw
        z_camera = math.sin(pitch_rad) * y_after_yaw + math.cos(pitch_rad) * z_after_yaw

        camera_distance = self._world_span_mm() * 3.5
        perspective = camera_distance / max(camera_distance - y_camera, camera_distance * 0.25)
        scale = self._scale_px_per_mm() * self.zoom
        screen_x = self.width() / 2.0 + x_camera * scale * perspective
        screen_y = self.height() / 2.0 - z_camera * scale * perspective
        return (screen_x, screen_y, y_camera)

    def _world_span_mm(self) -> float:
        return self.config.reach_radius_mm + TARGET_MARGIN_MM

    def _scale_px_per_mm(self) -> float:
        usable = min(self.width(), self.height()) - 56.0
        return usable / (2.0 * self._world_span_mm())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)

        self.current_angles_deg = (0.0, 0.0, 0.0)
        self.current_result: IKResult | None = None
        self.config = self._read_config_from_defaults()
        self.kinematics = MixedAxisArmKinematics(self.config)
        self.validator: URDFValidator | None = None

        self.view_3d = Orbit3DCanvas()
        self.top_view = ProjectionCanvas("Top View (X / Y)", "x", "y")
        self.side_view = ProjectionCanvas("Side View (X / Z)", "x", "z")
        self.view_3d.targetChanged3D.connect(self._on_3d_view_target_changed)
        self.top_view.targetChanged.connect(self._on_top_view_target_changed)
        self.side_view.targetChanged.connect(self._on_side_view_target_changed)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        layout.addWidget(self._build_controls(), 0)
        layout.addWidget(self._build_views(), 1)

        self.setCentralWidget(root)
        self.resize(1360, 820)
        self._refresh_models()
        self._reset_target_to_home()
        self._solve_current_target()

    def _build_controls(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        host.setFixedWidth(410)
        layout.addWidget(self._build_setup_box())
        layout.addWidget(self._build_lengths_box())
        layout.addWidget(self._build_target_box())
        layout.addWidget(self._build_results_box())
        layout.addStretch(1)
        return host

    def _build_views(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        projection_row = QWidget()
        projection_layout = QHBoxLayout(projection_row)
        projection_layout.setContentsMargins(0, 0, 0, 0)
        projection_layout.setSpacing(12)
        projection_layout.addWidget(self.top_view, 1)
        projection_layout.addWidget(self.side_view, 1)

        layout.addWidget(self.view_3d, 3)
        layout.addWidget(projection_row, 2)
        return host

    def _build_setup_box(self) -> QWidget:
        box = QGroupBox("Setup")
        form = QFormLayout(box)

        self.branch_combo = QComboBox()
        self.branch_combo.addItem("Auto", "auto")
        self.branch_combo.addItem("Positive yaw branch", "positive")
        self.branch_combo.addItem("Negative yaw branch", "negative")
        self.branch_combo.currentIndexChanged.connect(self._solve_current_target)
        form.addRow("Yaw branch", self.branch_combo)

        self.auto_solve_check = QCheckBox("Auto solve while editing")
        self.auto_solve_check.setChecked(True)
        form.addRow("", self.auto_solve_check)

        self.home_button = QPushButton("Reset to home target")
        self.home_button.clicked.connect(self._reset_target_to_home)
        form.addRow("", self.home_button)

        self.setup_summary = QLabel()
        self.setup_summary.setWordWrap(True)
        form.addRow("", self.setup_summary)
        return box

    def _build_lengths_box(self) -> QWidget:
        box = QGroupBox("Link lengths")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.length_spins: list[QDoubleSpinBox] = []
        for row, (caption, default_value) in enumerate(
            zip(
                ("L1 shoulder", "L2 yaw link", "L3 tool link", "Yaw plane height"),
                self.config.link_lengths_mm + (self.config.yaw_plane_height_mm,),
            )
        ):
            label = QLabel(caption)
            minimum = -500.0 if caption == "Yaw plane height" else 40.0
            spin = self._make_spin(minimum, 2000.0, 10.0, 1)
            spin.setSuffix(MM_SUFFIX)
            spin.setValue(default_value)
            spin.valueChanged.connect(self._on_geometry_changed)
            self.length_spins.append(spin)
            grid.addWidget(label, row, 0)
            grid.addWidget(spin, row, 1)

        return box

    def _build_target_box(self) -> QWidget:
        box = QGroupBox("Target position")
        form = QFormLayout(box)

        target_range = self.config.reach_radius_mm + TARGET_MARGIN_MM
        self.target_x_spin = self._make_spin(-target_range, target_range, 10.0, 1)
        self.target_x_spin.setSuffix(MM_SUFFIX)
        self.target_x_spin.valueChanged.connect(self._on_target_changed)
        form.addRow("Target X", self.target_x_spin)

        self.target_y_spin = self._make_spin(-target_range, target_range, 10.0, 1)
        self.target_y_spin.setSuffix(MM_SUFFIX)
        self.target_y_spin.valueChanged.connect(self._on_target_changed)
        form.addRow("Target Y", self.target_y_spin)

        self.target_z_spin = self._make_spin(-target_range, target_range, 10.0, 1)
        self.target_z_spin.setSuffix(MM_SUFFIX)
        self.target_z_spin.valueChanged.connect(self._on_target_changed)
        form.addRow("Target Z", self.target_z_spin)

        self.solve_button = QPushButton("Solve IK now")
        self.solve_button.clicked.connect(self._solve_current_target)
        form.addRow("", self.solve_button)
        return box

    def _build_results_box(self) -> QWidget:
        box = QGroupBox("Result")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.status_label.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.status_label.setStyleSheet("padding: 8px; background: #f8fafc;")
        layout.addWidget(self.status_label)

        self.joint_value_labels: list[QLabel] = []
        self.motor_value_labels: list[QLabel] = []
        for title in ("Joint A / Shoulder pitch", "Joint B / Yaw", "Joint C / Yaw"):
            line = QLabel(f"{title}: --")
            motor = QLabel("Motor target: --")
            self.joint_value_labels.append(line)
            self.motor_value_labels.append(motor)
            layout.addWidget(line)
            layout.addWidget(motor)

        self.tip_pose_label = QLabel("Tip: --")
        self.error_label = QLabel("Error: --")
        self.validation_label = QLabel("URDF validation: --")
        layout.addWidget(self.tip_pose_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.validation_label)

        self.export_box = QPlainTextEdit()
        self.export_box.setReadOnly(True)
        self.export_box.setPlaceholderText("Joint targets will appear here.")
        self.export_box.setFixedHeight(150)
        layout.addWidget(self.export_box)

        return box

    def _make_spin(self, minimum: float, maximum: float, step: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        return spin

    def _read_config_from_defaults(self) -> ArmConfig:
        return ArmConfig(
            link_lengths_mm=(280.0, 235.0, 150.0),
            yaw_plane_height_mm=40.0,
            joint_limits=(
                JointLimit(-60.0, 0.0),
                JointLimit(-75.0, 75.0),
                JointLimit(-90.0, 90.0),
            ),
            gear_ratios=(2.0, 2.0, 2.0),
        )

    def _refresh_models(self) -> None:
        self.config = ArmConfig(
            link_lengths_mm=tuple(spin.value() for spin in self.length_spins[:3]),
            yaw_plane_height_mm=self.length_spins[3].value(),
            joint_limits=self.config.joint_limits,
            gear_ratios=self.config.gear_ratios,
        )
        self.kinematics = MixedAxisArmKinematics(self.config)
        self.kinematics.write_demo_urdf(GENERATED_URDF_PATH)
        self.validator = URDFValidator(GENERATED_URDF_PATH) if IKPY_AVAILABLE else None
        self._sync_target_ranges()
        self._update_setup_summary()

    def _update_setup_summary(self) -> None:
        self.setup_summary.setText(
            "Axis mapping:\n"
            "joint_a = shoulder pitch\n"
            "joint_b = yaw\n"
            "joint_c = yaw\n\n"
            f"Yaw plane height above joint_a: {self.config.yaw_plane_height_mm:.1f} mm\n"
            "3D preview: drag empty space to orbit, drag X/Y/Z handles to move the IK target, wheel to zoom\n"
            f"Generated URDF: {GENERATED_URDF_PATH}\n"
            f"ikpy validation: {'enabled' if self.validator else 'not available'}"
        )

    def _sync_target_ranges(self) -> None:
        target_range = self.config.reach_radius_mm + TARGET_MARGIN_MM
        for spin in (self.target_x_spin, self.target_y_spin, self.target_z_spin):
            spin.setRange(-target_range, target_range)

    def _reset_target_to_home(self) -> None:
        total_reach = sum(self.config.link_lengths_mm)
        self.target_x_spin.setValue(total_reach)
        self.target_y_spin.setValue(0.0)
        self.target_z_spin.setValue(self.config.yaw_plane_height_mm)
        if not self.auto_solve_check.isChecked():
            self._solve_current_target()

    def _on_geometry_changed(self) -> None:
        self._refresh_models()
        if self.auto_solve_check.isChecked():
            self._solve_current_target()

    def _on_target_changed(self) -> None:
        if self.auto_solve_check.isChecked():
            self._solve_current_target()

    def _on_top_view_target_changed(self, target_x_mm: float, target_y_mm: float) -> None:
        self.target_x_spin.setValue(target_x_mm)
        self.target_y_spin.setValue(target_y_mm)
        if not self.auto_solve_check.isChecked():
            self._solve_current_target()

    def _on_side_view_target_changed(self, target_x_mm: float, target_z_mm: float) -> None:
        self.target_x_spin.setValue(target_x_mm)
        self.target_z_spin.setValue(target_z_mm)
        if not self.auto_solve_check.isChecked():
            self._solve_current_target()

    def _on_3d_view_target_changed(self, target_x_mm: float, target_y_mm: float, target_z_mm: float) -> None:
        self.target_x_spin.setValue(target_x_mm)
        self.target_y_spin.setValue(target_y_mm)
        self.target_z_spin.setValue(target_z_mm)
        if not self.auto_solve_check.isChecked():
            self._solve_current_target()

    def _solve_current_target(self) -> None:
        result = self.kinematics.solve_analytic(
            target_xyz=(
                self.target_x_spin.value(),
                self.target_y_spin.value(),
                self.target_z_spin.value(),
            ),
            current_angles_deg=self.current_angles_deg,
            branch_mode=self.branch_combo.currentData(),
        )

        if result.success and result.angles_deg is not None and self.validator is not None:
            urdf_frames = self.validator.forward_frames_xyz(result.angles_deg)
            result.urdf_tip_error_mm = self._distance_mm(urdf_frames[-1], result.frames_xyz[-1])

        self.current_result = result
        if result.success and result.angles_deg is not None:
            self.current_angles_deg = result.angles_deg
        self._render_result()

    def _render_result(self) -> None:
        assert self.current_result is not None
        result = self.current_result
        points_xyz = result.points_xyz
        target_xyz = result.target_xyz

        self.top_view.update_scene(self.config, points_xyz, target_xyz)
        self.side_view.update_scene(self.config, points_xyz, target_xyz)
        self.view_3d.update_scene(self.config, points_xyz, target_xyz)

        if result.success and result.angles_deg is not None:
            motor_angles = self.kinematics.joint_to_motor_angles(result.angles_deg)
            for label, title, value in zip(
                self.joint_value_labels,
                ("Joint A / Shoulder pitch", "Joint B / Yaw", "Joint C / Yaw"),
                result.angles_deg,
            ):
                label.setText(f"{title}: {value:,.2f} deg")
            for label, value in zip(self.motor_value_labels, motor_angles):
                label.setText(f"Motor target: {value:,.2f} deg")

            tip_x_mm, tip_y_mm, tip_z_mm = result.frames_xyz[-1]
            self.tip_pose_label.setText(
                f"Tip: X={tip_x_mm:,.1f} mm, Y={tip_y_mm:,.1f} mm, Z={tip_z_mm:,.1f} mm"
            )
            self.error_label.setText(f"Position error: {result.position_error_mm:,.3f} mm")
            if result.urdf_tip_error_mm is None:
                self.validation_label.setText("URDF validation: unavailable")
            else:
                self.validation_label.setText(
                    f"URDF validation tip error: {result.urdf_tip_error_mm:,.6f} mm"
                )
            self.export_box.setPlainText(
                "{\n"
                f"  \"joint_a\": {result.angles_deg[0]:.3f},\n"
                f"  \"joint_b\": {result.angles_deg[1]:.3f},\n"
                f"  \"joint_c\": {result.angles_deg[2]:.3f}\n"
                "}\n\n"
                "# Motor-side targets with the current 2.0 gear ratio\n"
                "{\n"
                f"  \"joint_a_motor\": {motor_angles[0]:.3f},\n"
                f"  \"joint_b_motor\": {motor_angles[1]:.3f},\n"
                f"  \"joint_c_motor\": {motor_angles[2]:.3f}\n"
                "}\n\n"
                f"# Generated URDF\n# {GENERATED_URDF_PATH}"
            )
        else:
            for label, title in zip(
                self.joint_value_labels,
                ("Joint A / Shoulder pitch", "Joint B / Yaw", "Joint C / Yaw"),
            ):
                label.setText(f"{title}: --")
            for label in self.motor_value_labels:
                label.setText("Motor target: --")
            self.tip_pose_label.setText("Tip: no valid solution")
            self.error_label.setText("Position error: target unreachable or outside joint limits")
            self.validation_label.setText("URDF validation: --")
            self.export_box.clear()

        status_prefix = "OK" if result.success else "Needs adjustment"
        self.status_label.setText(f"{status_prefix} | {result.message}")

    @staticmethod
    def _distance_mm(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]
        dz = point_a[2] - point_b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
