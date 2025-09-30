# -*- coding: utf-8 -*-
import struct
import time
import math


# Communication types (CommType)
COMM_TYPE_GET_ID = 0x00  # Get device ID
COMM_TYPE_MOTION_CONTROL = 0x01  # Motion control command
COMM_TYPE_MOTOR_REQUEST = 0x02  # Motor status feedback
COMM_TYPE_MOTOR_ENABLE = 0x03  # Motor enable
COMM_TYPE_MOTOR_STOP = 0x04  # Motor stop
COMM_TYPE_SET_POS_ZERO = 0x06  # Set mechanical zero position
COMM_TYPE_CAN_ID = 0x07  # Change CAN ID
COMM_TYPE_CONTROL_MODE = 0x12  # Control mode (single parameter read/write)
COMM_TYPE_GET_SINGLE_PARA = 0x11  # Read single parameter
COMM_TYPE_SET_SINGLE_PARA = 0x12  # Set single parameter
COMM_TYPE_ERROR_FEEDBACK = 0x15  # Error feedback

# Master ID and default CAN ID (names correspond to Master_CAN_ID and CanID in the C++ version)
MASTER_CAN_ID = 0x00
DEFAULT_CAN_ID = 0x01

# Physical mapping ranges (correspond to P_MIN, P_MAX, V_MIN, V_MAX, KP_MIN, KP_MAX, KD_MIN, KD_MAX, T_MIN, T_MAX in C++)
P_MIN = -12.5
P_MAX = 12.5
V_MIN = -30.0
V_MAX = 30.0
KP_MIN = 0.0
KP_MAX = 500.0
KD_MIN = 0.0
KD_MAX = 5.0
T_MIN = -12.0
T_MAX = 12.0

# Position range limits (steps) (corresponds to MAX_P / MIN_P in C++)
MAX_P = 720
MIN_P = -720

# Parameter identifiers (correspond to #define values in the original C++ code)
RUN_MODE = 0x7005
IQ_REF = 0x7006
SPD_REF = 0x700A
LIMIT_TORQUE = 0x700B
CUR_KP = 0x7010
CUR_KI = 0x7011
CUR_FILT_GAIN = 0x7014
LOC_REF = 0x7016
LIMIT_SPD = 0x7017
LIMIT_CUR = 0x7018

# Measurement conversion constants (correspond to Gain_Angle, Bias_Angle, Gain_Speed, etc. in C++)
GAIN_ANGLE = 720.0 / 32767.0
BIAS_ANGLE = 0x8000
GAIN_SPEED = 30.0 / 32767.0
BIAS_SPEED = 0x8000
GAIN_TORQUE = 12.0 / 32767.0
BIAS_TORQUE = 0x8000
TEMP_GAIN = 0.1

# CAN extended frame flag
CAN_EFF_FLAG = 0x80000000

# π 常數
PI = math.pi


# ============================================
# 3. 工具函式：浮點數↔位元組、映射、計算 ExtID、封裝 data
# ============================================

def float_to_bytes(f: float) -> bytes:
    """Pack a 32-bit Python float into 4 big-endian bytes (like Float_to_Byte in C++).
    """
    return struct.pack('>f', f)


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Linearly map float x into the integer range [0, 2^bits - 1].

    This corresponds to the float_to_uint helper in the original C++ code.
    """
    if x < x_min:
        x = x_min
    elif x > x_max:
        x = x_max

    max_int = (1 << bits) - 1
    span = x_max - x_min
    if span == 0:
        return 0
    return int(round((x - x_min) * max_int / span))


def calc_extid(comm_type: int, msid: int, can_id: int) -> int:
    """Compute the 29-bit extended CAN ID (ExtID).

    Args:
        comm_type: 8-bit communication type
        msid: 16-bit master ID (Master_CAN_ID)
        can_id: 8-bit target motor CAN ID

    Returns:
        A 32-bit integer where the lower 29 bits represent the ExtID.
    """
    msid_l = msid & 0xFF
    msid_h = (msid >> 8) & 0xFF

    di_data = ((comm_type & 0xFFFFFFFF) << 24) | 0x00FFFFFF
    di_datab = ((msid_h & 0xFFFFFFFF) << 16) | 0xFF00FFFF
    di_datac = ((msid_l & 0xFFFFFFFF) << 8) | 0xFFFF00FF
    di_datad = (can_id & 0xFFFFFFFF) | 0xFFFFFF00

    return di_data & di_datab & di_datac & di_datad


def data_count_dcs(index: int, value, value_type: str) -> bytes:
    """Pack an 8-byte payload for a single-parameter write (data_count_dcs in C++).

    Args:
        index: 16-bit software parameter index
        value: numeric value to write
        value_type: 'f' for float (packed as 4 bytes, reversed order), 's' for 8-bit integer
    """
    buf = bytearray(8)
    # 前兩 bytes 儲存 index (低位在 buf[0], 高位在 buf[1])
    buf[0] = index & 0xFF
    buf[1] = (index >> 8) & 0xFF
    buf[2] = 0x00
    buf[3] = 0x00

    if value_type == 'f':
        fb = float_to_bytes(float(value))   # 4 bytes 大端
        # C++ 版本放 byte_ls[3], byte_ls[2], byte_ls[1], byte_ls[0]
        buf[4] = fb[3]
        buf[5] = fb[2]
        buf[6] = fb[1]
        buf[7] = fb[0]
    else:  # 's'
        buf[4] = int(value) & 0xFF
        buf[5] = 0x00
        buf[6] = 0x00
        buf[7] = 0x00

    return bytes(buf)


def data_count_zero() -> bytes:
    """Return eight zero bytes (equivalent to data_count_zero() in C++).
    """
    return bytes([0] * 8)


def send_can_frame(ser, frame_id: int, data_bytes: bytes):
    """Send a CAN frame via the serial CAN bridge.

    The bridge expects: 0xAA 0xE8 + 4-byte ID (little-endian) + 8-byte data + 0x55.

    Args:
        ser: open pyserial Serial instance
        frame_id: 32-bit unsigned frame ID
        data_bytes: must be exactly 8 bytes
    """
    if len(data_bytes) != 8:
        raise ValueError("data_bytes length must be exactly 8 bytes")

    # 4-byte ID (little-endian) as required by the bridge
    id_bytes = frame_id.to_bytes(4, byteorder='little', signed=False)
    packet = id_bytes + data_bytes

    # Wrap with bridge framing bytes and write
    packet = b'\xAA' + b'\xE8' + packet + b'\x55'
    ser.write(packet)

    # Wait 4 ms to match expected bridge timing (equivalent to delay(4) in C++)
    time.sleep(0.004)


def motor_enable(ser, motor_id: int = DEFAULT_CAN_ID):
    """Enable the motor with the specified CAN ID.
    """
    extid = calc_extid(COMM_TYPE_MOTOR_ENABLE, MASTER_CAN_ID, motor_id)

    # Print the computed extended ID for debugging
    print(f"Enabling motor with extid: {hex(extid)}")
    data = data_count_zero()
    send_can_frame(ser, extid, data)
    time.sleep(0.004)


def motor_mode(ser, motor_id: int, mode_type: int):
    """Set the motor control mode.

    Corresponds to the C++ sequence that writes the RUN_MODE parameter.

    mode_type values:
      0 = motion control mode
      1 = position mode
      2 = speed mode
      3 = current mode
    """
    extid = calc_extid(COMM_TYPE_CONTROL_MODE, MASTER_CAN_ID, motor_id)

    data = data_count_dcs(RUN_MODE, mode_type, 's')
    send_can_frame(ser, extid, data)
    time.sleep(0.004)


def motor_yk(ser, motor_id: int,
             torque: float,
             mech_pos: float,
             speed: float,
             kp: float,
             kd: float):
    """Send a motion control command (equivalent to send_control_command in C++).

    The constructed ExtID and payload encode torque, motor ID, position,
    speed, Kp and Kd according to the original firmware protocol.
    """
    # 1) Build 29-bit ExtID: set bits 28..24 then torque (bits 23..8) and motor ID (bits 7..0)
    id_ext = (1 << 24)  # bits28..24 = 1
    id_ext |= (float_to_uint(torque, T_MIN, T_MAX, 16) << 8)  # bits23..8
    id_ext |= (motor_id & 0xFF)  # bits7..0

    # 2) Build 8-byte data payload
    buf = bytearray(8)

    # Byte0..1: position [-4π, +4π] -> uint16
    pu = float_to_uint(mech_pos, -4.0 * PI, 4.0 * PI, 16)
    buf[0] = (pu >> 8) & 0xFF
    buf[1] = pu & 0xFF

    # Byte2..3: speed [-30, +30] -> uint16
    vu = float_to_uint(speed, V_MIN, V_MAX, 16)
    buf[2] = (vu >> 8) & 0xFF
    buf[3] = vu & 0xFF

    # Byte4..5: Kp [0, 500] -> uint16
    kpu = float_to_uint(kp, KP_MIN, KP_MAX, 16)
    buf[4] = (kpu >> 8) & 0xFF
    buf[5] = kpu & 0xFF

    # Byte6..7: Kd [0, 5] -> uint16
    kdu = float_to_uint(kd, KD_MIN, KD_MAX, 16)
    buf[6] = (kdu >> 8) & 0xFF
    buf[7] = kdu & 0xFF

    data = bytes(buf)
    send_can_frame(ser, id_ext, data)

    time.sleep(0.004)


def motor_pos_zero(ser, motor_id: int = DEFAULT_CAN_ID):
    """Command the motor to set its current position as mechanical zero.
    """
    extid = calc_extid(COMM_TYPE_SET_POS_ZERO, MASTER_CAN_ID, motor_id)
    buf = bytearray(8)
    buf[0] = 1
    data = bytes(buf)
    send_can_frame(ser, extid, data)
    time.sleep(0.004)


def uint16_to_float(value, min_val, max_val):
    """Map a 16-bit unsigned integer in [0..65535] to a float in [min_val..max_val].

    Uses a linear mapping: min_val + (max_val - min_val) * (value / 65535.0)
    """
    return min_val + (max_val - min_val) * (value / 65535.0)


def read_current_position(ser):
    """Read one CAN frame from the serial bridge and return motor position.

    Expects the bridge framing: 0xAA, 0xE8, 4-byte ID (little-endian), 8-byte data, 0x55.
    Parses the data payload and returns (current_rad, current_deg) on success,
    or None if the frame is invalid or not a feedback frame.
    """

    # Read start-of-packet marker
    header = ser.read(1)
    if header != b'\xAA':
        # Not the expected start; ignore
        return None

    # Consume the bridge info byte (unused)
    _ = ser.read(1)

    # Read 4-byte CAN ID + 8 data bytes
    frame = ser.read(12)
    # Debug: print(f"Frame: {frame.hex(' ')}")
    if len(frame) < 12:
        # Incomplete frame
        return None

    # CAN ID is little-endian in the bridge frame
    can_id = int.from_bytes(frame[:4], byteorder='little', signed=False)
    data_bytes = frame[4:12]

    # Extract commType (assumed to be in bits 24..28 of CAN ID)
    commType = (can_id >> 24) & 0x1F
    if commType != 2:
        # Not a motor feedback frame; ignore
        return None

    # Combine two bytes into a 16-bit raw position
    rawPos = (data_bytes[0] << 8) | data_bytes[1]

    # Map to radians in the range [-12.56, +12.56]
    current_rad = uint16_to_float(rawPos, -12.56, 12.56)
    current_deg = current_rad * 180.0 / math.pi

    # Consume trailing end byte
    ser.read(1)
    return current_rad, current_deg
