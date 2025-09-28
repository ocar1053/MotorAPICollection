import threading
import serial
from cubemotorAK70Util import *

CAN_BAUDRATE = 2000000  # CAN 通訊速率
CAN_TIMEOUT = 5


def main():
    target_postion = 0
    # 1. open serial port
    try:
        ser = serial.Serial('COM11', CAN_BAUDRATE, timeout=CAN_TIMEOUT)
        time.sleep(0.1)  # wait for serial port to initialize

        frame = [
            0xaa, 0x55, 0x12,
            0x01,  # CAN speed: 1m
            0x02,  # extened frame
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
        print(f"fail：{e}")
        return

    # 2 make sure the serial port is open
    if not ser.is_open:
        print("Failed to open serial port.")
        return
    # open serial listener
    listener = SerialCanListener(ser)
    servo_mod_set_zero(ser, control_mode_id=5, motor_id=93)
    print("set to zero")

    time.sleep(1)  # wait for the motor to set to zero
    pos, spd, cur, temp, err, can_id = listener.get_status()
    print(
        f"Position: {pos}, Speed: {spd}, Current: {cur}, Temperature: {temp}, Error: {err}, can_id: {can_id} ")
    time.sleep(1)
    v = 0
    try:
        # speed control mode test
        # while 1:
        #     target_postion = 3600
        #     servo_mod_pos_speed(ser, control_mode_id=6,
        #                         motor_id=93, pos_deg=target_postion, speed_erpm=10000, rpa=30000)

        #     pos, spd, cur, temp, err, can_id = listener.get_status()
        #     print(
        #         f"Position: {pos}, Speed: {spd}, Current: {cur}, Temperature: {temp}, Error: {err}, CAN ID: {can_id}")
        #     time.sleep(1)

        # speed control mode test
        while 1:

            if v % 2 == 0:
                target_postion = 25.122
            else:
                target_postion = -25.122

            servo_mod_pos(ser, control_mode_id=4,
                          motor_id=93, pos_deg=target_postion)

            v += 1
            time.sleep(1)
            pos, spd, cur, temp, err, can_id = listener.get_status()
            print(
                f"Position: {pos}, Speed: {spd}, Current: {cur}, Temperature: {temp}, Error: {err}, CAN ID: {can_id}")
        print("Motor control completed.")

    finally:
        listener.close()
        if ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
