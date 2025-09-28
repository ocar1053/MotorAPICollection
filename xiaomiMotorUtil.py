# -*- coding: utf-8 -*-
import serial
import struct
import time
import math


# 通信類型 (CommType)
COMM_TYPE_GET_ID = 0x00  # 获取设备ID
COMM_TYPE_MOTION_CONTROL = 0x01  # 运控模式
COMM_TYPE_MOTOR_REQUEST = 0x02  # 电机状态反馈
COMM_TYPE_MOTOR_ENABLE = 0x03  # 电机使能
COMM_TYPE_MOTOR_STOP = 0x04  # 电机停止
COMM_TYPE_SET_POS_ZERO = 0x06  # 机械零位
COMM_TYPE_CAN_ID = 0x07  # 更改 CAN_ID
COMM_TYPE_CONTROL_MODE = 0x12  # 单参读写
COMM_TYPE_GET_SINGLE_PARA = 0x11  # 读取单个参数
COMM_TYPE_SET_SINGLE_PARA = 0x12  # 设定单参数
COMM_TYPE_ERROR_FEEDBACK = 0x15  # 错误反馈

# 主控 ID、預設 CAN ID（原 C++ 裡分別是 Master_CAN_ID 和 CanID）
MASTER_CAN_ID = 0x00
DEFAULT_CAN_ID = 0x01

# 物理量映射上下限（對應 C++ 裡的 P_MIN, P_MAX, V_MIN, V_MAX, KP_MIN, KP_MAX, KD_MIN, KD_MAX, T_MIN, T_MAX）
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

# 位置範圍(步數)上下限 (C++ 裡的 MAX_P / MIN_P)
MAX_P = 720
MIN_P = -720

# “参数读取宏定义” (C++ 裡的 #define Run_mode, Iq_Ref, Spd_Ref…)
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

# 量測量轉換常數 (C++ 裡的 Gain_Angle, Bias_Angle, Gain_Speed…)
GAIN_ANGLE = 720.0 / 32767.0
BIAS_ANGLE = 0x8000
GAIN_SPEED = 30.0 / 32767.0
BIAS_SPEED = 0x8000
GAIN_TORQUE = 12.0 / 32767.0
BIAS_TORQUE = 0x8000
TEMP_GAIN = 0.1

# CAN 擴展幀標誌位 (Extended Frame Flag)
CAN_EFF_FLAG = 0x80000000

# π 常數
PI = math.pi


# ============================================
# 3. 工具函式：浮點數↔位元組、映射、計算 ExtID、封裝 data
# ============================================

def float_to_bytes(f: float) -> bytes:
    """
    把 Python float (32-bit) 打包成 4 bytes (big-endian)，對應 C++ 裡的 Float_to_Byte。
    """
    return struct.pack('>f', f)


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """
    把浮點數 x 線性映射到 [0, 2^bits - 1] 之間的整數。
    對應 C++ 裡的 float_to_uint。
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
    """
    計算 29-bit 的擴展 ID (ExtID)，對應 C++ 裡的 exid_count。
    comm_type: 8-bit
    msid: 16-bit (Master_CAN_ID)
    can_id: 8-bit (目標馬達 CAN ID)
    回傳 32-bit，低 29 位是真正的 ID。
    """
    msid_l = msid & 0xFF
    msid_h = (msid >> 8) & 0xFF

    di_data = ((comm_type & 0xFFFFFFFF) << 24) | 0x00FFFFFF
    di_datab = ((msid_h & 0xFFFFFFFF) << 16) | 0xFF00FFFF
    di_datac = ((msid_l & 0xFFFFFFFF) << 8) | 0xFFFF00FF
    di_datad = (can_id & 0xFFFFFFFF) | 0xFFFFFF00

    return di_data & di_datab & di_datac & di_datad


def data_count_dcs(index: int, value, value_type: str) -> bytes:
    """
    封裝「单参数写入」的 8 bytes Data 對應 C++ 裡的 data_count_dcs：
      index: 16-bit SW 指令
      value_type: 'f' 表示浮點數 → 先 pack 成 4 bytes，再倒序；'s' 表示 8-bit 整數
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
    """
    回傳 8 個 0，對應 C++ 版裡的 data_count_zero()。
    """
    return bytes([0] * 8)


def send_can_frame(ser, frame_id: int, data_bytes: bytes):
    """
    **「4 Byte ID + 8 Byte Data」,12 bytes)。**



    frame_id: 32-bit unsigned
    data_bytes: 長度必須是 8 個 bytes
    """
    if len(data_bytes) != 8:
        raise ValueError("data_bytes 長度必須為 8")

    # 4 Byte ID (big-endian)
    id_bytes = frame_id.to_bytes(4, byteorder='little', signed=False)
    packet = id_bytes + data_bytes
    # print   # Debug: 印出十六進位格式的封包

    packet = b'\xAA' + b'\xE8'+packet + b'\x55'

    ser.write(packet)

    time.sleep(0.004)  # 等待 4 毫秒，對應 C++ 裡的 delay(4)


def motor_enable(ser, motor_id: int = DEFAULT_CAN_ID):
    """
    enable motor

    """
    extid = calc_extid(COMM_TYPE_MOTOR_ENABLE, MASTER_CAN_ID, motor_id)

    # print hex extid
    print(f"Enabling motor with extid: {hex(extid)}")
    data = data_count_zero()
    send_can_frame(ser, extid, data)
    time.sleep(0.004)


def motor_mode(ser, motor_id: int, mode_type: int):
    """
    對應 C++:
      exid_count(0x12, Master_CAN_ID, CanID);
      canMsg.can_id = ExtId | CAN_EFF_FLAG;
      data_count_dcs(0x7005, type, 's');
      mcp2515.sendMessage(...);
    mode_type: 0=运控模式,1=位置模式,2=速度模式,3=电流模式
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
    """
    C++ 裡的 send_control_command 對應到這裡 (有時命名為 motor_yk)：
      CommType=0x01 (MotionControl)，bit28~24=1
      bit23~8 = torque 映射到 [0..65535]
      bit7~0  = motor_id
      Data[0..1] = mech_pos ([-4π, +4π] → uint16)
      Data[2..3] = speed    ([-30, +30] → uint16)
      Data[4..5] = kp       ([0, 500] → uint16)
      Data[6..7] = kd       ([0,   5] → uint16)
    """
    # —— 1️⃣ 計算擴展 ID —— (29-bit)
    id_ext = (1 << 24)  # bit28~24 = 1
    id_ext |= (float_to_uint(torque, T_MIN, T_MAX, 16) << 8)  # bit23~8
    id_ext |= (motor_id & 0xFF)                             # bit7~0

    # —— 2️⃣ 填 8 Byte Data ——
    buf = bytearray(8)

    # Byte0~1: 位置 [-4π, +4π] → uint16
    pu = float_to_uint(mech_pos, -4.0 * PI, 4.0 * PI, 16)
    buf[0] = (pu >> 8) & 0xFF
    buf[1] = pu & 0xFF

    # Byte2~3: 速度 [-30, +30] → uint16
    vu = float_to_uint(speed, V_MIN, V_MAX, 16)
    buf[2] = (vu >> 8) & 0xFF
    buf[3] = vu & 0xFF

    # Byte4~5: Kp [0, 500] → uint16
    kpu = float_to_uint(kp, KP_MIN, KP_MAX, 16)
    buf[4] = (kpu >> 8) & 0xFF
    buf[5] = kpu & 0xFF

    # Byte6~7: Kd [0, 5] → uint16
    kdu = float_to_uint(kd, KD_MIN, KD_MAX, 16)
    buf[6] = (kdu >> 8) & 0xFF
    buf[7] = kdu & 0xFF

    data = bytes(buf)
    send_can_frame(ser, id_ext, data)

    time.sleep(0.004)


def motor_pos_zero(ser, motor_id: int = DEFAULT_CAN_ID):

    extid = calc_extid(COMM_TYPE_SET_POS_ZERO, MASTER_CAN_ID, motor_id)
    buf = bytearray(8)
    buf[0] = 1
    data = bytes(buf)
    send_can_frame(ser, extid, data)
    time.sleep(0.004)


def uint16_to_float(value, min_val, max_val):
    """
    把 0..65535 的 uint16 對應到 [min_val, max_val] 的浮點
    原 C++: float currentPosition = uint16_to_float(rawPos, -12.56f, 12.56f, 16);
    這裡假設是線性對應 (range-mapping)：
        float = min_val + (max_val - min_val) * (value / 65535.0)
    """
    return min_val + (max_val - min_val) * (value / 65535.0)


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

    data_bytes = frame[4:12]

    # Extract commType (假設通訊類型在 CAN ID 的 bit24..bit28)
    commType = (can_id >> 24) & 0x1F

    if commType != 2:
        # 不是我們要的 "feedback frame"，就忽略
        return None

    # Combine high/low 8 bits → 16 位 rawPos
    rawPos = (data_bytes[0] << 8) | data_bytes[1]

    # 轉成弧度範圍 [-12.56, +12.56]
    current_rad = uint16_to_float(rawPos, -12.56, 12.56)
    # 轉成角度
    current_deg = current_rad * 180.0 / math.pi

    # read end
    ser.read(1)  # 讀取最後的結尾 byte
    return current_rad, current_deg
