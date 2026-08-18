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

            return p.device

    for p in ports:

        pass
    return None

def test_triggerbox(port=None):
    if port is None:
        port = find_triggerbox_port()
    if port is None:

        return False

    try:
        ser = serial.Serial(port, timeout=1)

    except Exception as e:

        return False


    codes = [11, 12, 21, 22, 31, 99, 0]
    for code in codes:
        ser.write(bytes([code]))

        time.sleep(0.5)
        ser.write(bytes([0]))                 
        time.sleep(0.5)

    ser.close()

    return True

if __name__ == '__main__':
    port = sys.argv[1] if len(sys.argv) > 1 else None
    test_triggerbox(port)