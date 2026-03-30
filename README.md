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
```

## Troubleshooting

- If you receive no telemetry, verify wiring, CAN bitrate, motor ID, and
  converter mode.
- If framing errors appear, confirm the bridge packet format matches the
  expected `0xAA ... 0x55` framing.
