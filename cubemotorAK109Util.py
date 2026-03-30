# -*- coding: utf-8 -*-
import serial
import struct
import time
import threading

PROTO_POS_DEG_MIN = -36000.0
PROTO_POS_DEG_MAX = 36000.0
SPEED_ERPM_MIN = -327680
SPEED_ERPM_MAX = 327670
RPA_MIN = 0
RPA_MAX = 327670
SAFE_POS_TURN_LIMIT_DEG = 360.0
SAFE_POS_DEG_MIN = -SAFE_POS_TURN_LIMIT_DEG
SAFE_POS_DEG_MAX = SAFE_POS_TURN_LIMIT_DEG
BRAKE_CURRENT_MIN_A = -60.0
BRAKE_CURRENT_MAX_A = 60.0


class SerialCanListener:

    def __init__(self, ser, can_id: int = 0x00):
        """
        Initialize the SerialCanListener with the specified serial port and baud rate.
        """
        self.ser = ser
        self._lock = threading.Lock()
        self.can_id = can_id  # Default CAN ID
        self._pos = 0
        self._spd = 0
        self._cur = 0
        self._cur_temp = 0
        self._error = 0
        self._running = True
        self._motor_dic = {}
        self._th = threading.Thread(target=self._read_loop, daemon=True)
        

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
            #print(f"can id: {can_id & 0xFF}")
         
            # self.can_id = can_id & 0xFF
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
                self._motor_dic[can_id & 0xFF] = {
                    "position": motr_pos,
                    "speed": motr_spd,
                    "current": motr_cur,
                    "temperature": temp,
                    "error": err
                }
            self.ser.read(1)  # Consume the trailing end byte (0x55)

    def get_status(self):
        """
        Get the current status of the motor.
        Returns a tuple of (position, speed, current, temperature, error).
        """
        with self._lock:
            return dict(self._motor_dic)

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

 
    ser.write(packet)

    time.sleep(0.004)


def enforce_safe_position_limit(pos_deg: float) -> float:
    """Reject target positions beyond one mechanical turn."""
    if not SAFE_POS_DEG_MIN <= pos_deg <= SAFE_POS_DEG_MAX:
        raise ValueError(
            f"Target position {pos_deg} deg exceeds one-turn safety limit "
            f"({SAFE_POS_DEG_MIN} to {SAFE_POS_DEG_MAX} deg)."
        )
    return pos_deg


def clamp_brake_current(brake_current_a: float) -> float:
    """Clamp current-brake requests to the protocol-supported range."""
    return min(max(brake_current_a, BRAKE_CURRENT_MIN_A), BRAKE_CURRENT_MAX_A)


def servo_mod_set_zero(ser, control_mode_id: int, motor_id: int):
    """
    Set the motor's permanent zero position. control id = 5
    """

    data = bytes([0x01])

    id_bytes = calc_extid(control_mode_id, motor_id).to_bytes(
        4, byteorder='little', signed=False)
    packet = id_bytes + data

    # E1 for 1 data length
    packet = b'\xAA' + b'\xE1'+packet + b'\x55'

    ser.write(packet)
    ser.flush()
    # Give the controller a short window to commit zero-offset handling.
    time.sleep(0.03)


def servo_mod_pos_speed(ser, control_mode_id: int, motor_id: int, pos_deg: float, speed_erpm: int, rpa: int):
    """
    set the motor to position mode with speed. control id = 6

    pos_deg arg accept -36000 to 36000 (degree)
    speed_erpm arg accept -327680~-327680 (erpm)
    rpa arg accept accept 0~327670 1 unit is equal to 10 electrical speed/s².
    """
    pos_deg = enforce_safe_position_limit(pos_deg)
    speed_erpm = min(max(speed_erpm, SPEED_ERPM_MIN), SPEED_ERPM_MAX)
    rpa = min(max(rpa, RPA_MIN), RPA_MAX)

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

    pos_deg = enforce_safe_position_limit(pos_deg)

    # strech to
    val = int(pos_deg * 10000.0)

    # to big-endian 4 bytes
    buffer = bytearray(4)
    buffer[0] = (val >> 24) & 0xFF
    buffer[1] = (val >> 16) & 0xFF
    buffer[2] = (val >> 8) & 0xFF
    buffer[3] = val & 0xFF

    send_can_frame(ser, motor_id, buffer, control_mode_id)


def servo_mod_current_brake(ser, control_mode_id: int, motor_id: int, brake_current_a: float):
    """Apply current-brake mode. control id = 2.

    The protocol encodes brake current as int32 where 1000 represents 1 A.
    """
    brake_current_a = clamp_brake_current(brake_current_a)
    current_int = int(brake_current_a * 1000.0)

    buffer = bytearray(4)
    buffer[0] = (current_int >> 24) & 0xFF
    buffer[1] = (current_int >> 16) & 0xFF
    buffer[2] = (current_int >> 8) & 0xFF
    buffer[3] = current_int & 0xFF

    send_can_frame(ser, motor_id, buffer, control_mode_id)
