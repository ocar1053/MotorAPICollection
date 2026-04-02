# CubeMotor Motor API Collection

A small Python API collection and GUI for controlling CubeMotor AK10-9 and
AK70 series motors through a Waveshare USB-to-CAN bridge. The repository
focuses on servo position, position-speed, zeroing, and current-brake
commands used by the current tools and GUI.

> IMPORTANT: Read the motor user manual and verify each motor's CAN ID before
> connecting or operating the motor. Sending commands to the wrong device can
> be dangerous.

## Supported hardware

- CubeMotor AK10-9 series
- CubeMotor AK70 series
- CubeMotor AK60 series
- Waveshare USB-to-CAN (USB-CAN) converter

## Supported software

- Python 3.12
- pyserial (listed in `requirement.txt`)

## Quick start (Windows PowerShell)

1. Install Python 3.12 and verify it is available on `PATH`:

```powershell
python --version
```

2. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirement.txt
```

## Configuration

- Update the example scripts to use the correct COM port for your
  USB-to-CAN bridge.
- Ensure the bridge is in normal operating mode with the expected CAN bitrate
  before sending commands.

## Safety notes

- The scripts send real motor commands. Secure or unload the motor before
  running position tests.
- Start with small target angles and conservative motion settings.
- Monitor current and temperature during testing.

## Run an example

```powershell
# AK70 example
python cubemotorAK70.py

# Servo hold test
python cubemotor_servo_hold_test.py --family ak70 --port COM12 --motor-id 93 --target-deg 10

# GUI
python robot_arm_gui.py

# MediaPipe webcam preview only
python robot_arm_mediapipe_control.py --camera 0

# MediaPipe with two webcams and real robot follow
python robot_arm_mediapipe_control.py --camera 0 --camera 1 --arm --port COM12

# RealSense D455 preview only
python robot_arm_realsense_control.py --side right

# RealSense D455 with real robot follow
python robot_arm_realsense_control.py --side right --arm --port COM12
```

## MediaPipe arm follow

- `robot_arm_mediapipe_control.py` is intentionally separate from the GUI.
- Default mapping is:
  - `joint_a` <- shoulder pitch from the tracked shoulder side
  - `joint_b` <- shoulder yaw from the same shoulder side
  - `joint_c` <- elbow yaw from the tracked elbow side
- Default tracked side is the left arm, matching MediaPipe landmarks `11 -> 13 -> 15`.
- Press `c` to recalibrate your current body pose as the robot neutral pose.
- Press `q` or `Esc` to stop.
- Start without `--arm` first so you can verify the on-screen angles before moving the real hardware.

## RealSense D455 arm follow

- `robot_arm_realsense_control.py` is a separate D455-specific script.
- It still uses MediaPipe Pose for 2D landmark detection, but then samples aligned
  RealSense depth around the shoulder, elbow, and wrist landmarks to recover
  camera-space 3D points.
- This is especially useful for `joint_c` / motor ID `93`, where a plain RGB
  webcam can struggle with forearm yaw.
- Start with preview-only first:

```powershell
python robot_arm_realsense_control.py --side right
```

- To capture a reusable neutral calibration file, stand in your normal exhibit pose
  and save the first valid calibration:

```powershell
python robot_arm_realsense_control.py --side right --save-calibration realsense_calibration.json
```

- On later runs, reuse that saved calibration so startup skips first-pose
  auto-calibration:

```powershell
python robot_arm_realsense_control.py --side right --load-calibration realsense_calibration.json
```

- If you want to load an existing calibration and keep updating the same file when
  you press `n` / `p` / `b` / `c`, pass both flags:

```powershell
python robot_arm_realsense_control.py --side right --load-calibration realsense_calibration.json --save-calibration realsense_calibration.json
```

- Then arm the real robot only after the depth-based angles look stable:

```powershell
python robot_arm_realsense_control.py --side right --arm --port COM12
```

## Troubleshooting

- If you receive no telemetry, verify wiring, CAN bitrate, motor ID, and
  converter mode.
- If framing errors appear, confirm the bridge packet format matches the
  expected `0xAA ... 0x55` framing.
