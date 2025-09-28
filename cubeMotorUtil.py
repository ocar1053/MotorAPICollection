# -*- coding: utf-8 -*-
import serial
import struct
import time
import math
from dataclasses import dataclass

CONTROL_MODE_MIT_ID = 0x08
DEFAULT_MOTOR_CAN_ID = 0x01

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
    position: float
    velocity: float
    torque:   float  # torque(current)
    kp:       float
    kd:       float


def calc_extid(control_mode_id: int, motor_id: int):
    """
    cal 29-bit extended ID for CAN frame.

    28-8 bit control mode ID
    7-0  bit motor ID
    """

    ex_id = (control_mode_id << 8) | (motor_id & 0xFF)

    return ex_id


def send_can_frame(ser, control_mode_id: int, motor_id: int, data_bytes: bytes):
    """
    **「4 Byte ID + 8 Byte Data」,12 bytes)。**
    frame_id: 32-bit unsigned

    """

    # 4 Byte ID (little)
    id_bytes = calc_extid(control_mode_id, motor_id).to_bytes(
        4, byteorder='little', signed=False)

    packet = id_bytes + data_bytes
    # print   # Debug: 印出十六進位格式的封包

    packet = b'\xAA' + b'\xE8'+packet + b'\x55'
    # print(packet.hex(" "))  # Debug: 印出十六進位格式的封包
    ser.write(packet)

    time.sleep(0.004)  # 等待 4 毫秒，對應 C++ 裡的 delay(4)


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """
    Convert a float value to an unsigned integer representation within a specified range and bit width.
    """
    span = x_max - x_min

    if x < x_min:
        x = x_min
    elif x > x_max:
        x = x_max

    return int((x - x_min) * ((1 << bits) / span))


def mit_pos(ser):
    # bytearray id 0x 00 00 08 01
    # bytearray 00 06 66 7F FF 8F 57 FF
    id_ = bytearray([0x01, 0x08, 0x00, 0x00])
    data = bytearray([0x00, 0x06, 0x66, 0x7F, 0xFF, 0x8F, 0x57, 0xFF])
    packet = id_ + data
    packet = bytearray([0xAA, 0xE8]) + packet + bytearray([0x55])

    # Debug: Print the packet in hex format
    print("Sending packet:", packet.hex(" "))
    ser.write(packet)


def pack_cmd_mit_mode_data(position: float, velocity: float, torque: float, kp: float, kd: float) -> bytearray:

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
    # DATA[0] = KP 高 8 位
    data[0] = (kp_int >> 4) & 0xFF
    # DATA[1] bits7-4：KP 低 4 位； bits3-0：KD 高 4 位
    data[1] = ((kp_int & 0xF) << 4) | ((kd_int >> 8) & 0xF)
    # DATA[2] = KD 低 8 位
    data[2] = kd_int & 0xFF
    # DATA[3] = 位置高 8 位
    data[3] = (p_int >> 8) & 0xFF
    # DATA[4] = 位置低 8 位
    data[4] = p_int & 0xFF
    # DATA[5] = 速度高 8 位
    data[5] = (v_int >> 4) & 0xFF
    # DATA[6] bits7-4：速度低 4 位；bits3-0：电流(扭矩)高 4 位
    data[6] = ((v_int & 0xF) << 4) | ((torque_int >> 8) & 0xF)
    # DATA[7] = 电流(扭矩)低 8 位
    data[7] = torque_int & 0xFF

    return data


def mit_mode(ser, config: PositionModeConfig, motor_id: int = DEFAULT_MOTOR_CAN_ID, control_mode_id: int = CONTROL_MODE_MIT_ID):
    """
    Set the motor to position mode.
    """

    data = pack_cmd_mit_mode_data(
        position=config.position,
        velocity=config.velocity,
        torque=config.torque,
        kp=config.kp,
        kd=config.kd
    )

    send_can_frame(ser, control_mode_id, motor_id, data)


def set_to_zero_postion(ser, motor_id: int = DEFAULT_MOTOR_CAN_ID, control_mode_id: int = CONTROL_MODE_MIT_ID):
    """
    Set the motor to zero position.
    """
    data = bytes([0x00])
    send_can_frame(ser, control_mode_id, motor_id, data)


def read_current_position(ser):
    """
    從 serial (pyserial) 讀取一個 CAN frame，並回傳位置（弧度、角度）。
    假設封包格式是：4 bytes CAN ID（big-endian）+ 8 bytes data
    如果錯誤或 commType != 2 就回傳 None。

    回傳值格式：(current_rad, current_deg) 或 None
    """
    # 嘗試讀 12 個 byte：4 bytes CAN ID + 8 bytes data[]
    # 如果實際封包長度不同，須自行調整這裡的讀取長度

    # fisrt read wave
    header = ser.read(1)

    if header != b'\xAA':

        return None  # 如果不是預期的開頭，就忽略這個封包

    info = ser.read(1)

    frame = ser.read(12)
    # print(f"Frame: {frame.hex(' ')}")  # Debug: 印出整個 frame 的十六進位格式
    if len(frame) < 12:
        # 如果讀不到足夠的位元組，就不做處理
        return None

    # 解析 CAN ID (4 bytes, big-endian)
    can_id = int.from_bytes(frame[:4], byteorder='little', signed=False)

    # print(can_id.hex(" "))  # Debug: 印出 CAN ID 的十六進位格式
    data_bytes = frame[4:12]
    pos_int, spd_int, cur_int = struct.unpack('>hhh', data_bytes[0:6])

    temp, error = struct.unpack('>bB', data_bytes[6:8])

    # 位置 (度)
    motor_pos = pos_int * 0.1
    # 速度 (rpm)
    motor_spd = spd_int * 10.0
    # 電流 (A)
    motor_cur = cur_int * 0.01
    # 溫度 (℃)
    motor_temp = temp
    # 報錯碼
    motor_error = error

    print(motor_pos, motor_spd, motor_cur, motor_temp, motor_error)

    # read end
    ser.read(1)  # 讀取最後的結尾 byte
    # return current_rad, current_deg
