"""
Test communication with TriggerBox
Run on WINDOWS before experiment
"""
import serial, time, sys

def find_triggerbox_port():
    """Find TriggerBox COM port"""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if 'TriggerBox' in p.description or 'triggerbox' in p.description.lower():
            print(f"OK: Found TriggerBox on {p.device}: {p.description}")
            return p.device
    print("Available ports:")
    for p in ports:
        print(f"  {p.device}: {p.description}")
    return None

def test_triggerbox(port=None):
    if port is None:
        port = find_triggerbox_port()
    if port is None:
        print("ERROR: TriggerBox not found. Check USB cable.")
        return False

    try:
        ser = serial.Serial(port, timeout=1)
        print(f"OK: Port {port} opened")
    except Exception as e:
        print(f"ERROR: Cannot open {port}: {e}")
        return False

    print("\nTest: sending codes 11, 12, 21, 22, 31 (1 Hz)...")
    print("Check in BrainVision Recorder if markers appear in the trigger channel.")
    codes = [11, 12, 21, 22, 31, 99, 0]
    for code in codes:
        ser.write(bytes([code]))
        print(f"  Sent: {code}")
        time.sleep(0.5)
        ser.write(bytes([0]))  # reset to zero
        time.sleep(0.5)

    ser.close()
    print("\nOK: Test completed. Check markers in Recorder.")
    return True

if __name__ == '__main__':
    port = sys.argv[1] if len(sys.argv) > 1 else None
    test_triggerbox(port)