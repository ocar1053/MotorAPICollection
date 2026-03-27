import threading
import serial
from cubemotorAK109Util import *

CAN_BAUDRATE = 2000000  # CAN bus bitrate (bits/s)
CAN_TIMEOUT = 5

stop_flag = False
def read_status(listener: SerialCanListener):
    while not stop_flag:
        status_dic = listener.get_status()
        for motor_id, status in status_dic.items():
            print(
                f"Motor ID: {motor_id}, Position: {status['position']}, Speed: {status['speed']}, Current: {status['current']}, Temperature: {status['temperature']}, Error: {status['error']}")
        time.sleep(0.5)
def main():
    target_postion = 0
    # 1. open serial port
    try:
        ser = serial.Serial('COM18', CAN_BAUDRATE, timeout=CAN_TIMEOUT)
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
    servo_mod_set_zero(ser, control_mode_id=5, motor_id=0)
    
    servo_mod_set_zero(ser, control_mode_id=5, motor_id=1)
    print("set to zero")

    
    time.sleep(1)  # wait for the motor to set to zero
    #open thread to read status
    thread = threading.Thread(target=read_status, args=(listener,), daemon=True)
    thread.start()
    
    v = 0
    try:
        # speed control mode test
        
        # while 1:
        #     target_postion = 360
        #     servo_mod_pos_speed(ser, control_mode_id=6,
        #                         motor_id=0, pos_deg=target_postion, speed_erpm=5000, rpa=30000)
        #     print(f"Set target position to: {target_postion} degrees")
        #     # pos, spd, cur, temp, err, can_id = listener.get_status()
        #     print(
        #         f"Position: {pos}, Speed: {spd}, Current: {cur}, Temperature: {temp}, Error: {err}, CAN ID: {can_id}")
        #     # time.sleep(1)

        # speed control mode test
        target_postion = 0
        while 1:

            target_postion += 10
            if target_postion > 360:
                break

            servo_mod_pos(ser, control_mode_id=4,
                          motor_id=0, pos_deg=target_postion)
            servo_mod_pos(ser, control_mode_id=4,
                          motor_id=1, pos_deg=target_postion*-1)
            print(f"Set target position to: {target_postion} degrees")
            v += 1
            time.sleep(1)
        print("Motor control completed.")

    finally:
        listener.close()
        stop_flag = True
        if ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
