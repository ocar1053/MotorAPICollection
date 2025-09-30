# xiaomi_motor — CubeMotor AK60 / AK70 API Collection

A small Python API collection and example scripts for controlling CubeMotor
Xiaomi AK60 and AK70 series motors using a Waveshare USB-to-CAN (USB-CAN)
converter. The code provides helpers to build and send CAN frames over a
serial bridge and implements the motor payload formats used by the firmware
(MIT-style position commands and other control modes).

> IMPORTANT: Read the motor user manual and verify each motor's CAN ID before
> connecting or operating the motor. Sending commands to the wrong device can
> be dangerous.

## Supported hardware

- CubeMotor AK60 series
- CubeMotor AK70 series
- Waveshare USB-to-CAN (USB-CAN) converter (bridge packet format: `0xAA 0xE8 + ID + data + 0x55`)

## Supported software

- Python 3.12
- pyserial (listed in `requirement.txt`)


## Quick start (Windows PowerShell)

1. Install Python 3.12 and verify it is available on `PATH`:

```powershe
python --version
```

2. Create and activate a virtual environment (PowerShell):

```powershe
python -m venv venv
\venv\Scripts\Activate.ps1
```



## Configuration

- Edit the example scripts to set the correct serial port and CAN settings.
  Examples: `SERIAL_PORT = 'COM6'` in `cubemotorAK60.py`, and `COM11` in
  `cubemotorAK70.py` — change these to match the COM port shown in Device
  Manager for your Waveshare USB-CAN device.
- The scripts send a configuration frame to the Waveshare bridge on startup —
  ensure the bridge is in normal operating mode and the desired CAN bitrate is
  selected.

## Safety notes

- The example scripts send real motor commands. Secure or unload the motor
  before running sweeps.
- Start with low PID gains and small target positions to avoid abrupt or
  strong motion. Monitor current and temperature during tests.
- These scripts assume specific bridge framing and motor firmware payloads —
  they are not intended to be general-purpose CAN libraries.

## Run an example

```powershell
# AK60 sweep
python cubemotorAK60.py

# AK70 test
python cubemotorAK70.py

# Two-motor demo
python xiaomi.py
```

## Troubleshooting

- If you receive no telemetry, verify wiring, CAN bitrate, motor ID, and the
  converter mode (not loopback).
- If parsing or framing errors appear, capture a serial hex dump and confirm
  the bridge framing bytes (`0xAA 0xE8 ... 0x55`).

## Contributing

Small documentation or usability improvements are welcome. Please keep
changes that affect motor safety or electrical behavior carefully reviewed.

