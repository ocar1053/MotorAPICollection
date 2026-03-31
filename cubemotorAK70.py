import time

import serial

from cubemotorAK70Util import *

CAN_BAUDRATE = 2000000  # CAN bus bitrate (bits/s)
CAN_TIMEOUT = 5
SERIAL_PORT = "COM12"
MOTOR_ID = 93
TARGET_SEQUENCE_DEG = [0, 10, 20, 10, 0, -10, -20, -10, 0]
POSITION_SPEED_ERPM = 2000
POSITION_ACCEL_ERPM_S = 5000


def main():
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
    # Zero-on-start disabled to avoid overwriting the stored origin.
    # servo_mod_set_zero(ser, control_mode_id=5, motor_id=MOTOR_ID)
    # print("set to zero")

    time.sleep(1)  # wait for the motor to set to zero
    pos, spd, cur, temp, err, can_id = listener.get_status()
    print(
        f"Position: {pos}, Speed: {spd}, Current: {cur}, "
        f"Temperature: {temp}, Error: {err}, can_id: {can_id} "
    )
    time.sleep(1)

    try:
        for target_position in TARGET_SEQUENCE_DEG:
            servo_mod_pos_speed(
                ser,
                control_mode_id=6,
                motor_id=MOTOR_ID,
                pos_deg=target_position,
                speed_erpm=POSITION_SPEED_ERPM,
                rpa=POSITION_ACCEL_ERPM_S,
            )
            time.sleep(1.5)
            pos, spd, cur, temp, err, can_id = listener.get_status()
            print(
                f"Target: {target_position} deg, Position: {pos}, Speed: {spd}, "
                f"Current: {cur}, Temperature: {temp}, Error: {err}, CAN ID: {can_id}"
            )
        print("Motor control completed.")

    finally:
        listener.close()
        if ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
