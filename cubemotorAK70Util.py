# -*- coding: utf-8 -*-
import serial
import struct
import time
import math
from dataclasses import dataclass
import threading

P_MIN = -12.5
P_MAX = 12.5
V_MIN = -50.0
V_MAX = 50.0
T_MIN = -25.0
T_MAX = 25.0
KP_MIN = 0.0
KP_MAX = 500.0
KD_MIN = 0.0
KD_MAX = 5.0


class SerialCanListener:

    def __init__(self, ser):
        """
        Initialize the SerialCanListener with the specified serial port and baud rate.
        """
        self.ser = ser
        self._lock = threading.Lock()
        self.can_id = 0x00  # Default CAN ID
        self._pos = 0
        self._spd = 0
        self._cur = 0
        self._cur_temp = 0
        self._error = 0
        self._running = True
        self._th = threading.Thread(target=self._read_loop, daemon=True)
        self._th.start()

    def _read_loop(self):
        """
        Continuously read from the serial port and update the motor status.
        """
        while self._running:

            header = self.ser.read(1)

            if header != b'\xAA':

                continue

            info = self.ser.read(1)

            frame = self.ser.read(12)

            if len(frame) < 12:

                continue

            can_id = int.from_bytes(
                frame[:4], byteorder='little', signed=False)
            # Extract the motor ID from the CAN ID (low byte)
            self.can_id = can_id & 0xFF
            data_bytes = frame[4:12]

            pos_int = int.from_bytes(
                data_bytes[0:2], byteorder='big', signed=True)

            spd_int = int.from_bytes(
                data_bytes[2:4], byteorder='big', signed=True)
            cur_int = int.from_bytes(
                data_bytes[4:6], byteorder='big', signed=True)
            temp = int.from_bytes(
                data_bytes[6:7], byteorder='big', signed=True)
            err = data_bytes[7]  # uint8

            # strech encoder value according to the spec
            motr_pos = float(pos_int * 0.1)
            motr_spd = float(spd_int * 10)
            motr_cur = float(cur_int * 0.01)

            with self._lock:
                self._pos = motr_pos
                self._spd = motr_spd
                self._cur = motr_cur
                self._cur_temp = temp
                self._error = err
            self.ser.read(1)  # Consume the trailing end byte (0x55)

    def get_status(self):
        """
        Get the current status of the motor.
        Returns a tuple of (position, speed, current, temperature, error).
        """
        with self._lock:
            return (self._pos, self._spd, self._cur, self._cur_temp, self._error, self.can_id)

    def close(self):
        """
        Close the thread and stop reading from the serial port.
        """
        self._running = False
        self._th.join()


def calc_extid(control_mode_id: int, motor_id: int):
    """Calculate the 29-bit extended CAN ID used by the bridge.

    The ID format places the control mode ID in bits 8..28 and the motor ID
    in the lowest 8 bits (bits 0..7).
    """

    ex_id = (control_mode_id << 8) | (motor_id & 0xFF)

    return ex_id


def send_can_frame(ser, motor_id: int, data_bytes: bytes, control_mode_id: int):
    """
    Send a CAN frame to the motor with the specified motor ID and data bytes.

    """

    id_bytes = calc_extid(control_mode_id, motor_id).to_bytes(
        4, byteorder='little', signed=False)
    packet = id_bytes + data_bytes

    if len(data_bytes) == 8:
        # E8 for 8 data length
        packet = b'\xAA' + b'\xE8' + packet + b'\x55'
    if len(data_bytes) == 4:
        # E4 for 4 data length
        packet = b'\xAA' + b'\xE4' + packet + b'\x55'

    # print(packet.hex(" "))
    ser.write(packet)

    time.sleep(0.004)


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

    # DATA[0] = pos high 8
    data[0] = (p_int >> 8) & 0xFF
    # DATA[1] = pos low 8
    data[1] = p_int & 0xFF

    # DATA[2] = velocity high 8 bits
    data[2] = (v_int >> 8) & 0xFF
    # DATA[3] bits7-4: velocity low 4 bits; bits3-0: KP high 4 bits
    data[3] = ((v_int & 0xF) << 4) | ((kp_int >> 8) & 0xF)

    # DATA[4] = KP low 8 bits
    data[4] = kp_int & 0xFF

    # DATA[5] = KD high 8 bits
    data[5] = (kd_int >> 4) & 0xFF
    # DATA[6] bits7-4: KD low 4 bits; bits3-0: torque high 4 bits
    data[6] = ((kd_int & 0xF) << 4) | ((torque_int >> 8) & 0xF)

    # DATA[7] = torque low 8 bits
    data[7] = torque_int & 0xFF

    return data


def servo_mod_set_zero(ser, control_mode_id: int, motor_id: int):
    """
    Set the motor to zero position. control id = 5
    """

    data = bytes([0x00])

    id_bytes = calc_extid(control_mode_id, motor_id).to_bytes(
        4, byteorder='little', signed=False)
    packet = id_bytes + data

    # E1 for 1 data length
    packet = b'\xAA' + b'\xE1'+packet + b'\x55'
    print(packet.hex(" "))
    ser.write(packet)


def servo_mod_pos_speed(ser, control_mode_id: int, motor_id: int, pos_deg: float, speed_erpm: int, rpa: int):
    """
    set the motor to position mode with speed. control id = 6

    pos_deg arg accept -36000 to 36000 (degree)
    speed_erpm arg accept -327680~-327680 (erpm)
    rpa arg accept accept 0~327670 1 unit is equal to 10 electrical speed/s².
    """
    pos_int = int(pos_deg * 10000.0)
    spd_int = int(speed_erpm / 10.0)
    rpa_int = int(rpa / 10.0)

    buf = bytearray(8)

    # pos_int (32 bit)
    buf[0] = (pos_int >> 24) & 0xFF
    buf[1] = (pos_int >> 16) & 0xFF
    buf[2] = (pos_int >> 8) & 0xFF
    buf[3] = pos_int & 0xFF

    # spd_int (16 bit)
    buf[4] = (spd_int >> 8) & 0xFF
    buf[5] = spd_int & 0xFF
    # rpa_int (16 bit)
    buf[6] = (rpa_int >> 8) & 0xFF
    buf[7] = rpa_int & 0xFF

    send_can_frame(ser, motor_id, buf, control_mode_id)


def servo_mod_pos(ser,  control_mode_id: int, motor_id: int, pos_deg: float = 0.0):
    """
    Set the motor to position mode. control id  = 4
    pos_deg arg accept -36000 to 36000 (degree) 

    """

    # strech to
    val = int(pos_deg * 10000.0)

    # to big-endian 4 bytes
    buffer = bytearray(4)
    buffer[0] = (val >> 24) & 0xFF
    buffer[1] = (val >> 16) & 0xFF
    buffer[2] = (val >> 8) & 0xFF
    buffer[3] = val & 0xFF

    send_can_frame(ser, motor_id, buffer, control_mode_id)
