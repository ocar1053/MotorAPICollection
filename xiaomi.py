import threading
import serial
import time
from xiaomiMotorUtil import *

CAN_BAUDRATE = 2000000
CAN_TIMEOUT = 0.1


def position_reader_thread(ser, stop_event):
    """Background thread that continuously reads motor position and prints it.

    The loop repeatedly calls ``read_current_position(ser)`` and prints the
    telemetry when available. The thread exits when ``stop_event`` is set by
    the main thread.
    """
    while not stop_event.is_set():
        result = read_current_position(ser)
        if result is not None:
            rad, deg = result
            print(
                f"[Reader] Current angle (rad) = {rad:.6f}, (deg) = {deg:.2f}")
        # Small sleep to avoid busy-waiting; adjust as needed
        time.sleep(0.05)

    print("[Reader] Stop signal received, exiting reader thread.")


def main():
    target_postion = 0
    # 1. open serial port
    try:
        ser = serial.Serial('COM5', CAN_BAUDRATE, timeout=CAN_TIMEOUT)
        time.sleep(0.1)  # allow the serial port to initialize

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
        print(f"Failed: {e}")
        return

    # 2 make sure the serial port is open
    if not ser.is_open:
        print("Failed to open serial port.")
        return
    # 2. Create an Event to notify the reader thread to stop
    stop_event = threading.Event()

    # 3. Create and start the position_reader_thread
    reader_thread = threading.Thread(
        target=position_reader_thread,
        args=(ser, stop_event),
        daemon=True,  # daemon=True makes the thread exit automatically when the program ends
    )
    reader_thread.start()
    # bussiness logic
    try:
        # # motor1
        motor_enable(ser, 127)
        motor_pos_zero(ser, 127)
        motor_mode(ser, 127, 0)
        time.sleep(0.1)  # wait for motor to initialize

        while True:
            target_postion = 0
            while target_postion <= 1.56:
                motor_yk(
                    ser, 127, 1, target_postion, 1, 10, 0.5)
                target_postion += 0.03
                time.sleep(0.0003)
            # back to 0 -= 0.01
            target_postion = 1.56
            while target_postion >= 0.2:
                motor_yk(
                    ser, 127, 1, target_postion, 1, 10, 0.5)
                target_postion -= 0.03
                time.sleep(0.0003)
            time.sleep(1)
 
        print("Motor control completed.")
        # Keep the main thread alive; replace with your application loop
        while True:
            time.sleep(1)
    finally:
        stop_event.set()        
        reader_thread.join()    
        if ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
