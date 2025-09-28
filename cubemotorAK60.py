import threading
import serial
import time
from cubeMotorUtilAK60 import *


CAN_BAUDRATE = 2000000
CAN_TIMEOUT = 0.1
SERIAL_PORT = 'COM6'


def position_reader_thread(ser, stop_event):
    """Background thread that continuously reads motor telemetry.

    This function repeatedly calls ``read_current_position(ser)`` and prints
    the telemetry returned by that function. The loop exits when ``stop_event``
    is set by the main thread.
    """

    while not stop_event.is_set():
        read_current_position(ser)

    print("[Reader] Stop signal received, exiting reader thread.")


def main():
    """Open the serial bridge, start a reader thread, and run a simple motion test.

    The script configures the serial-to-CAN bridge, starts a background reader
    thread to print telemetry, then sends MIT position commands to the motor
    in a rising and falling position sweep.
    """

    # 1. open serial port and configure bridge
    try:
        ser = serial.Serial(SERIAL_PORT, CAN_BAUDRATE, timeout=CAN_TIMEOUT)
        time.sleep(0.1)  # allow the serial port to initialize

        # Build bridge configuration frame (according to waveshare usb-can converter protocol)
        frame = [
            0xaa, 0x55, 0x12,
            0x01,  # CAN speed: 1m
            0x02,  # extended frame
            0x00, 0x00, 0x00, 0x00,  # Filter ID
            0x00, 0x00, 0x00, 0x00,  # Mask ID
            0x00,  # Normal mode
            0x01, 0x00, 0x00, 0x00, 0x00  # Reserved
        ]
        checksum = sum(frame[2:19]) & 0xFF
        frame.append(checksum)
        ser.write(bytearray(frame))

        time.sleep(0.1)  # wait for the configuration frame to be sent
    except Exception as e:
        print(f"Failed to open or configure serial port: {e}")
        return

    # 2. ensure the serial port opened successfully
    if not ser.is_open:
        print("Failed to open serial port.")
        return

    # 3. create and start the position-reader thread
    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=position_reader_thread,
        args=(ser, stop_event),
        daemon=True,  # daemon=True makes the thread exit automatically when the program ends
    )
    reader_thread.start()

    # Business logic: send MIT position commands
    try:

        target_position = 0

        # Motor 1: initialize position to zero
        time.sleep(0.1)  # short delay to ensure the motor/bridge is ready
        mit_par = PositionModeConfig(
            position=target_position,
            velocity=0.0,
            torque=0.0,
            kp=2.0,
            kd=2.0,
        )

        mit_mode(ser, mit_par, motor_id=1, control_mode_id=8)

        time.sleep(1)  # wait for motor to initialize

        # sweep up to 12.56 rad
        while target_position <= 12.56:
            mit_par.position = target_position
            mit_mode(ser, mit_par, motor_id=1, control_mode_id=8)

            target_position += 0.03
            time.sleep(0.01)

        # sweep back toward
        time.sleep(1)
        target_position -= 0.03
        while target_position >= 0.2:
            mit_par.position = target_position
            mit_mode(ser, mit_par, motor_id=1, control_mode_id=8)

            target_position -= 0.03
            time.sleep(0.01)

        time.sleep(1)
        print("Motor control completed.")

        # Keep the main thread alive (replace with your application loop)
        while True:
            time.sleep(1)

    finally:
        # Signal the reader thread to stop and wait for it to finish
        stop_event.set()
        reader_thread.join()
        if ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
