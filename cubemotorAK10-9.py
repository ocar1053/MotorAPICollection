import threading
import time

import serial

from cubemotorAK109Util import *

CAN_BAUDRATE = 2000000  # CAN bus bitrate (bits/s)
CAN_TIMEOUT = 5
SERIAL_PORT = "COM12"
MOTOR_IDS = (0, 1)
TARGET_SEQUENCE_DEG = [0, 10, 20, 30, 20, 10, 0]
POSITION_SPEED_ERPM = 2000
POSITION_ACCEL_ERPM_S = 5000

stop_flag = False


def read_status(listener: SerialCanListener):
    while not stop_flag:
        status_dic = listener.get_status()
        for motor_id, status in sorted(status_dic.items()):
            print(
                f"Motor ID: {motor_id}, Position: {status['position']}, "
                f"Speed: {status['speed']}, Current: {status['current']}, "
                f"Temperature: {status['temperature']}, Error: {status['error']}"
            )
        time.sleep(0.5)


def main():
    global stop_flag
    stop_flag = False

    # 1. open serial port
    try:
        ser = serial.Serial(SERIAL_PORT, CAN_BAUDRATE, timeout=CAN_TIMEOUT)
        time.sleep(0.1)  # wait for serial port to initialize

        frame = [
            0xaa, 0x55, 0x12,
            0x01,  # CAN speed: 1M (1 megabit/s)
            0x02,  # extended frame
            0x00, 0x00, 0x00, 0x00,  # Filter ID
            0x00, 0x00, 0x00, 0x00,  # Mask ID
            0x00,  # Normal mode
            0x01, 0x00, 0x00, 0x00, 0x00  # Reserved
        ]
        checksum = sum(frame[2:19]) & 0xFF
        frame.append(checksum)
        ser.write(bytearray(frame))

        time.sleep(0.1)  # wait for the frame to be sent
    except Exception as e:
        print(f"Failed: {e}")
        return

    # 2 make sure the serial port is open
    if not ser.is_open:
        print("Failed to open serial port.")
        return

    # start serial listener
    listener = SerialCanListener(ser)
    listener._th.start()
    for motor_id in MOTOR_IDS:
        servo_mod_set_zero(ser, control_mode_id=5, motor_id=motor_id)
    print("set to zero")

    time.sleep(1)  # wait for the motor to set to zero
    thread = threading.Thread(target=read_status, args=(listener,), daemon=True)
    thread.start()

    try:
        for target_position in TARGET_SEQUENCE_DEG:
            servo_mod_pos_speed(
                ser,
                control_mode_id=6,
                motor_id=MOTOR_IDS[0],
                pos_deg=target_position,
                speed_erpm=POSITION_SPEED_ERPM,
                rpa=POSITION_ACCEL_ERPM_S,
            )
            servo_mod_pos_speed(
                ser,
                control_mode_id=6,
                motor_id=MOTOR_IDS[1],
                pos_deg=-target_position,
                speed_erpm=POSITION_SPEED_ERPM,
                rpa=POSITION_ACCEL_ERPM_S,
            )
            print(
                f"Set target position to: {target_position} degrees "
                f"(speed={POSITION_SPEED_ERPM} erpm, rpa={POSITION_ACCEL_ERPM_S})"
            )
            time.sleep(1.5)
        print("Motor control completed.")

    finally:
        stop_flag = True
        thread.join(timeout=1.0)
        listener.close()
        if ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
