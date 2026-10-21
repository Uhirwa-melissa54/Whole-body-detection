"""USB angle-only protocol for the user-supplied OK/HB stepper sketch."""
import argparse
import math
import time


class AngleController:
    def __init__(self, mode="pointer", invert=False, minimum=15, maximum=165,
                 deadzone=0.08, gain=1.0, speed=45.0, smoothing=0.15):
        self.mode, self.invert = mode, invert
        self.minimum, self.maximum = minimum, maximum
        self.deadzone, self.gain = deadzone, gain
        self.speed, self.smoothing = speed, smoothing
        self.angle = self.filtered = 90.0
        self.last_time = None
        self.paused = False

    def reset(self):
        self.angle = self.filtered = 90.0

    def update(self, error, now):
        dt = 0.0 if self.last_time is None else min(max(now - self.last_time, 0), 0.1)
        self.last_time = now
        if self.paused or error is None:
            self.angle = self.filtered  # freeze immediately on a detection gap
            return int(round(self.filtered))
        error = max(-1.0, min(1.0, float(error)))
        if self.invert:
            error = -error
        if self.mode == "pointer":
            self.angle = 90.0 if abs(error) <= self.deadzone else 90.0 + error * 90.0 * self.gain
        elif abs(error) > self.deadzone:
            self.angle += self.speed * self.gain * error * dt
        self.angle = max(self.minimum, min(self.maximum, self.angle))
        alpha = 1.0 if self.smoothing == 0 else 1.0 - math.exp(-dt / self.smoothing)
        self.filtered += alpha * (self.angle - self.filtered)
        return int(round(self.filtered))


class USBLink:
    def __init__(self, port, baud=115200):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("Install USB support: python -m pip install -r requirements-hardware.txt") from exc
        from serial.tools import list_ports
        available = list(list_ports.comports())
        if port.upper() not in {p.device.upper() for p in available}:
            listing = ", ".join(f"{p.device}: {p.description}" for p in available) or "none"
            raise RuntimeError(f"ESP8266 port {port} is absent. Available ports: {listing}. Reconnect the board with a USB data cable and run --list-ports again; use the CP210x port, not the generic Communications Port. To test only the camera: python -m src.face_tracking --any-face --camera-only")
        try:
            self.serial = serial.Serial(port, baud, timeout=0, write_timeout=0.25)
        except serial.SerialException as exc:
            raise RuntimeError(f"Cannot open {port}: {exc}. Close Arduino Serial Monitor and other programs using this port.") from exc
        self.buffer = bytearray()
        self.status = "Waiting for board"
        self.last_angle = None
        self.last_send = 0.0
        self.ready = False
        print("Waiting for Arduino startup sweep; do not move the needle while powered.")
        deadline = time.monotonic() + 20.0
        try:
            while time.monotonic() < deadline:
                self.poll()
                if self.ready:
                    break
                time.sleep(0.05)
            if not self.ready:
                raise RuntimeError("No angle-only firmware reply. Upload the supplied USB sketch, use 115200 baud, and close Serial Monitor. Expected 'Waiting for Python tracker angles', OK: or HB:.")
        except Exception:
            self.serial.close()
            raise

    def command(self, text):
        text = str(text).strip()
        if not text.isdigit() or not 0 <= int(text) <= 180:
            raise ValueError("This Arduino sketch accepts only integer angles 0..180; no HOME/OFF/STOP commands")
        self.serial.write((text + "\n").encode("ascii"))

    def poll(self):
        count = min(self.serial.in_waiting, 2048)
        if count:
            self.buffer.extend(self.serial.read(count))
        while b"\n" in self.buffer:
            line, _, rest = self.buffer.partition(b"\n")
            self.buffer = bytearray(rest)
            message = line.decode("ascii", errors="replace").strip()
            if message.startswith("Waiting for Python tracker angles"):
                self.ready = True
                self.status = "Board ready at startup center"
            elif message.startswith("OK:"):
                fields = message.split(":")
                if len(fields) == 3 and all(field.lstrip("-").isdigit() for field in fields[1:]):
                    self.ready = True
                    self.status = f"Accepted angle={fields[1]} target_steps={fields[2]}"
            elif message.startswith("HB:"):
                fields = message.split(":")
                if len(fields) == 3 and fields[1].isdigit() and "/" in fields[2]:
                    self.ready = True
                    self.status = f"Board angle={fields[1]} steps={fields[2]}"
            elif message.startswith("Calibrating") or message.startswith("Tachometer centered"):
                self.status = message
        if len(self.buffer) > 2048:
            self.buffer.clear()
        return self.status

    def send_angle(self, angle, now):
        if (angle != self.last_angle and now - self.last_send >= 0.04) or now - self.last_send >= 0.3:
            self.command(str(angle))
            self.last_angle, self.last_send = angle, now

    def close(self):
        # Firmware has no STOP command; sending text is not supported.
        self.serial.close()


def main():
    parser = argparse.ArgumentParser(description="List USB ports or test ESP8266 stepper without a camera")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--port")
    args = parser.parse_args()
    if args.list_ports:
        from serial.tools import list_ports
        for port in list_ports.comports():
            print(f"{port.device}: {port.description}")
        return
    if not args.port:
        parser.error("use --list-ports or --port COMx")
    link = USBLink(args.port)
    print("Enter an angle 15..165, or Q to quit. Arduino performs its own startup sweep.")
    print("Position needle at 0 degrees BEFORE opening USB; startup sweeps across the full configured travel.")
    try:
        while True:
            print(link.poll())
            command = input("ESP8266> ").strip().upper()
            if command == "Q":
                break
            if command.isdigit() and 15 <= int(command) <= 165:
                link.command(str(int(command)))
            else:
                print("Use one of the commands above.")
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                link.poll()
                time.sleep(0.02)
    finally:
        link.close()


if __name__ == "__main__":
    main()
