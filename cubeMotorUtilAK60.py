"""Utility functions for communicating with AK60-series motors over a CAN-over-serial bridge.

This module provides helpers to build and send CAN frames through a waveshare usb canbus
converter (packet format: 0xAA 0xE8 + 4-byte ID + 8-byte data + 0x55) and to
encode the MIT position control payload expected by the motor firmware.

"""

import serial
import struct
import time
import math
from dataclasses import dataclass


# Control/motor identifiers
CONTROL_MODE_MIT_ID = 0x08
DEFAULT_MOTOR_CAN_ID = 0x01


# Allowed ranges for the MIT position-control fields
P_MIN = -12.56
P_MAX = 12.56
V_MIN = -60.0
V_MAX = 60.0
T_MIN = -12.0
T_MAX = 12.0
KP_MIN = 0.0
KP_MAX = 500.0
KD_MIN = 0.0
KD_MAX = 5.0


@dataclass
class PositionModeConfig:
    """Configuration container for MiT position-mode commands.

    Attributes:
        position: desired position in radians
        velocity: desired velocity in rad/s
        torque: feed-forward torque/current
        kp: proportional gain
        kd: derivative gain
    """
    position: float
    velocity: float
    torque: float  # torque/current feed-forward
    kp: float
    kd: float


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Map a floating-point value into an unsigned integer with given bit width.

    The value is clamped to [x_min, x_max] and then linearly scaled to the
    integer range [0, 2^bits).

    Args:
        x: value to encode
        x_min: minimum allowed value
        x_max: maximum allowed value
        bits: number of bits for the unsigned integer

    Returns:
        Integer representation of the clamped and scaled value.
    """

    span = x_max - x_min

    if x < x_min:
        x = x_min
    elif x > x_max:
        x = x_max

    return int((x - x_min) * ((1 << bits) / span))


def calc_extid(control_mode_id: int, motor_id: int) -> int:
    """Compute the 29-bit extended CAN ID used by the motor bridge.

    The upper bits (bits 8..28) hold the control mode identifier; the lower
    8 bits hold the motor ID.

    Args:
        control_mode_id: control mode identifier (e.g. MIT = 0x08)
        motor_id: motor CAN ID (0..255)

    Returns:
        A 32-bit integer containing the extended ID (packed in the same way
        the bridge expects it when converted to little-endian bytes).
    """

    ex_id = (control_mode_id << 8) | (motor_id & 0xFF)
    return ex_id


def send_can_frame(ser, control_mode_id: int, motor_id: int, data_bytes: bytes):
    """Send a CAN frame through the serial CAN bridge.

    Packet format used by the bridge: 0xAA 0xE8 + 4-byte ID (little-endian) +
    8-byte data + 0x55. The function sleeps 4 ms after writing to match the
    timing expected by the hardware.

    Args:
        ser: an open pyserial Serial instance
        control_mode_id: control mode identifier (upper bytes of the extended ID)
        motor_id: motor CAN ID (lower 8 bits of extended ID)
        data_bytes: 8 bytes of payload data
    """

    # 4-byte ID in little-endian order as required by the bridge
    id_bytes = calc_extid(control_mode_id, motor_id).to_bytes(
        4, byteorder='little', signed=False)

    packet = id_bytes + data_bytes
    # Wrap the ID+data with the bridge framing bytes
    packet = b'\xAA' + b'\xE8' + packet + b'\x55'
    # Debug: print the full packet in hex if needed
    # print(packet.hex(" "))
    ser.write(packet)

    # Wait 4 ms to match expected bridge timing (equivalent to delay(4) in C++)
    time.sleep(0.004)


def pack_cmd_mit_mode_data(position: float, velocity: float, torque: float, kp: float, kd: float) -> bytearray:
    """Pack a position-control command payload for MIT mode.

    The MIT payload is 8 bytes and contains KP, KD, position, velocity and
    torque/current packed with specific bit widths (KP:12, KD:12, P:16, V:12,
    T:12). Inputs are clamped to the predefined ranges before encoding.

    Returns:
        A bytearray of length 8 containing the encoded command.
    """

    p_des = min(max(position, P_MIN), P_MAX)
    v_des = min(max(velocity, V_MIN), V_MAX)
    kp_val = min(max(kp, KP_MIN), KP_MAX)
    kd_val = min(max(kd, KD_MIN), KD_MAX)
    t_ff = min(max(torque, T_MIN), T_MAX)

    p_int = float_to_uint(p_des, P_MIN, P_MAX, 16)
    v_int = float_to_uint(v_des, V_MIN, V_MAX, 12)
    kp_int = float_to_uint(kp_val, KP_MIN, KP_MAX, 12)
    kd_int = float_to_uint(kd_val, KD_MIN, KD_MAX, 12)
    torque_int = float_to_uint(t_ff, T_MIN, T_MAX, 12)

    data = bytearray(8)
    # DATA[0] = KP high 8 bits
    data[0] = (kp_int >> 4) & 0xFF
    # DATA[1] bits7-4: KP low 4 bits; bits3-0: KD high 4 bits
    data[1] = ((kp_int & 0xF) << 4) | ((kd_int >> 8) & 0xF)
    # DATA[2] = KD low 8 bits
    data[2] = kd_int & 0xFF
    # DATA[3] = position high 8 bits
    data[3] = (p_int >> 8) & 0xFF
    # DATA[4] = position low 8 bits
    data[4] = p_int & 0xFF
    # DATA[5] = velocity high 8 bits
    data[5] = (v_int >> 4) & 0xFF
    # DATA[6] bits7-4: velocity low 4 bits; bits3-0: torque high 4 bits
    data[6] = ((v_int & 0xF) << 4) | ((torque_int >> 8) & 0xF)
    # DATA[7] = torque low 8 bits
    data[7] = torque_int & 0xFF

    return data


def mit_mode(ser, config: PositionModeConfig, motor_id: int = DEFAULT_MOTOR_CAN_ID, control_mode_id: int = CONTROL_MODE_MIT_ID):
    """Send a position-mode (MIT) command to a motor.

    The function encodes the provided configuration and uses the serial
    bridge to transmit the packet.
    """

    data = pack_cmd_mit_mode_data(
        position=config.position,
        velocity=config.velocity,
        torque=config.torque,
        kp=config.kp,
        kd=config.kd,
    )

    send_can_frame(ser, control_mode_id, motor_id, data)


def read_current_position(ser):
    """Read one CAN frame from the serial bridge and parse motor telemetry.

    The bridge packet framing is expected to be: 0xAA, 0xE8, 4-byte ID,
    8-byte data, 0x55. This function reads the serial stream, validates the
    header byte, and parses the 8 data bytes into position, speed, current,
    temperature and error code. If the incoming bytes are invalid or
    incomplete, None is returned.

    Returns:
        A tuple (position_deg, speed_rpm, current_A, temp_C, error_code) or
        None when parsing fails or insufficient bytes are available.
    """

    # Read start-of-packet marker
    header = ser.read(1)

    if header != b'\xAA':
        # Not the expected packet start; ignore
        return None

    # Next byte is the bridge info/command byte (consumed but not used here)
    info = ser.read(1)

    # Read 4-byte CAN ID + 8 data bytes
    frame = ser.read(12)
    # Debug: print the full frame in hex
    # print(f"Frame: {frame.hex(' ')}")
    if len(frame) < 12:
        # Incomplete frame
        return None

    # Extract the 8 data bytes and decode fields
    data_bytes = frame[4:12]
    pos_int, spd_int, cur_int = struct.unpack('>hhh', data_bytes[0:6])

    temp, error = struct.unpack('>bB', data_bytes[6:8])

    # Convert raw values to human units
    motor_pos = pos_int * 0.1      # position in degrees
    motor_spd = spd_int * 10.0     # speed in rpm
    motor_cur = cur_int * 0.01     # current in A
    motor_temp = temp              # temperature in degC
    motor_error = error            # error/status code

    print(motor_pos, motor_spd, motor_cur, motor_temp, motor_error)

    # Consume the trailing end byte (0x55)
    ser.read(1)
    return motor_pos, motor_spd, motor_cur, motor_temp, motor_error
