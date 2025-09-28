import threading
import serial
from xiaomiMotorUtil import *

CAN_BAUDRATE = 2000000
CAN_TIMEOUT = 0.1


def position_reader_thread(ser, stop_event):
    """
    這個函式會持續在 background thread 執行，不斷呼叫 read_current_position，
    並把讀到的位置印出來。若 stop_event 被設定，就跳出迴圈結束執行緒。
    """
    while not stop_event.is_set():
        result = read_current_position(ser)
        if result is not None:
            rad, deg = result
            print(f"[Reader] 目前角度 (rad) = {rad:.6f}, (deg) = {deg:.2f}")
        # 間隔時間可依實際需求調整——這裡先 sleep 0.05 秒左右

    print("[Reader] 收到停止訊號，結束讀取執行緒。")


def main():
    target_postion = 0
    # 1. open serial port
    try:
        ser = serial.Serial('COM11', CAN_BAUDRATE, timeout=CAN_TIMEOUT)
        time.sleep(0.1)  # wait for serial port to initialize

        frame = [
            0xaa, 0x55, 0x12,
            0x01,  # CAN speed: 500k = 0x03
            0x02,  # Standard frame
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
    # 2. 建立 Event 讓我們可以通知 reader thread 停止
    stop_event = threading.Event()

    # 3. 建立並啟動 position_reader_thread
    reader_thread = threading.Thread(
        target=position_reader_thread,
        args=(ser, stop_event),
        daemon=True   # daemon=True 表示程式結束時自動結束此執行緒
    )
    reader_thread.start()
    # bussiness logic
    try:
        # # motor1
        motor_enable(ser, 1)
        motor_pos_zero(ser, 1)
        motor_mode(ser, 1, 0)
        time.sleep(0.1)  # wait for motor to initialize

        motor_enable(ser, 2)
        motor_pos_zero(ser, 2)
        motor_mode(ser, 2, 0)
        target_postion = 0
        while target_postion <= 1.56:
            motor_yk(
                ser, 1, 1, target_postion, 1, 10, 0.5)
            motor_yk(
                ser, 2, 1, target_postion, 1, 10, 0.5)
            target_postion += 0.03
            time.sleep(0.0003)
        # back to 0 -= 0.01
        target_postion = 1.56
        while target_postion >= 0.2:
            motor_yk(
                ser, 1, 1, target_postion, 1, 10, 0.5)
            motor_yk(
                ser, 2, 1, target_postion, 1, 10, 0.5)
            target_postion -= 0.03
            time.sleep(0.0003)
        time.sleep(1)
        # repeat the above process
        target_postion = 0.2
        while target_postion <= 1.56:
            motor_yk(
                ser, 1, 1, target_postion, 1, 10, 0.5)
            motor_yk(
                ser, 2, 1, target_postion, 1, 10, 0.5)
            target_postion += 0.03
            time.sleep(0.0003)
        # back to 0 -= 0.01
        target_postion = 1.56
        while target_postion >= 0.2:
            motor_yk(
                ser, 1, 1, target_postion, 1, 10, 0.5)
            motor_yk(
                ser, 2, 1, target_postion, 1, 10, 0.5)
            target_postion -= 0.03
            time.sleep(0.0003)

        time.sleep(1)
        target_postion = 0.2

        while target_postion <= 1.56:
            motor_yk(
                ser, 1, 1, target_postion, 1, 10, 0.5)
            motor_yk(
                ser, 2, 1, target_postion, 1, 10, 0.5)
            target_postion += 0.03
            time.sleep(0.0003)
        # back to 0 -= 0.01
        target_postion = 1.56
        while target_postion >= 0.2:
            motor_yk(
                ser, 1, 1, target_postion, 1, 10, 0.5)
            motor_yk(
                ser, 2, 1, target_postion, 1, 10, 0.5)
            target_postion -= 0.03
            time.sleep(0.0003)

        # target_postion = 0
        # while target_postion <= 1.56:
        #     motor_yk(
        #         ser, 2, 0, target_postion, 1, 10, 0.5)
        #     target_postion += 0.01
        #     time.sleep(0.01)
        print("Motor control completed.")
        while 1:
            pass
    finally:
        stop_event.set()        # 設定 Event，讓 position_reader_thread 跳出迴圈
        reader_thread.join()    # 等待讀取執行緒真正結束
        if ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
