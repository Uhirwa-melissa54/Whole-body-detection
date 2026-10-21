"""MQTT angle transport for the ESP8266 firmware used in this project."""
import argparse
import json
import threading
import time
import uuid

COMMAND_TOPIC = "face-centering/motor/command"
STATUS_TOPIC = "face-centering/motor/status"


class MQTTLink:
    def __init__(self, broker, port=1883, username=None, password=None):
        import paho.mqtt.client as mqtt
        self.mqtt = mqtt
        self.connected = threading.Event()
        self.status = "Waiting for ESP8266 telemetry"
        self.last_status = 0.0
        self.last_angle = None
        self.last_send = 0.0
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id="face-tracker-" + uuid.uuid4().hex[:10])
        if username:
            self.client.username_pw_set(username, password)
        self.client.on_connect = self._connect
        self.client.on_disconnect = self._disconnect
        self.client.on_message = self._message
        try:
            self.client.connect(broker, port, keepalive=15)
            self.client.loop_start()
            if not self.connected.wait(5):
                raise RuntimeError("MQTT connection was not accepted; check broker authentication.")
        except Exception as exc:
            self.client.disconnect()
            self.client.loop_stop()
            raise RuntimeError(f"Cannot connect to MQTT broker {broker}:{port}: {exc}. Start Mosquitto first; see MQTT_SETUP.md.") from exc

    def _connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(STATUS_TOPIC, qos=0)
            self.connected.set()
            self.last_angle = None
            self.last_send = 0.0
        else:
            self.status = f"Broker rejected connection: {reason_code}"

    def _disconnect(self, client, userdata, flags, reason_code, properties):
        self.connected.clear()
        self.status = "Broker disconnected; firmware watchdog stops motion"

    def _message(self, client, userdata, message):
        try:
            data = json.loads(message.payload.decode("utf-8"))
            if not isinstance(data, dict):
                return
            self.last_status = time.monotonic()
            self.status = (f"{data.get('status', '?')} "
                           f"steps={data.get('position_steps', '?')} target={data.get('target_steps', '?')} "
                           f"angle={data.get('angle', '?')}")
        except (ValueError, UnicodeError, TypeError):
            self.status = "Invalid board telemetry"

    def send_angle(self, angle, now):
        if not self.connected.is_set():
            raise RuntimeError("MQTT broker disconnected")
        angle = int(round(max(20.0, min(160.0, float(angle)))))
        if (angle == self.last_angle and now - self.last_send < 0.3):
            return None
        data = {"angle": angle}
        # Never retain movement/calibration commands for a reconnecting device.
        result = self.client.publish(COMMAND_TOPIC, json.dumps(data), qos=0, retain=False)
        if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError("MQTT command could not be published")
        self.last_angle, self.last_send = angle, now
        return result

    def poll(self):
        return self.status

    def close(self):
        self.client.disconnect()
        self.client.loop_stop()


def main():
    import os
    parser = argparse.ArgumentParser(description="MQTT stepper test without camera")
    parser.add_argument("--broker", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    args = parser.parse_args()
    link = MQTTLink(args.broker, args.mqtt_port,
                    os.getenv("FACE_MQTT_USERNAME"), os.getenv("FACE_MQTT_PASSWORD"))
    print("Enter angles 20..160, or Q. This firmware accepts only JSON angle commands.")
    try:
        while True:
            print(link.poll())
            text = input("MQTT> ").strip().upper()
            if text == "Q":
                break
            try:
                angle = int(text)
                if not 20 <= angle <= 160:
                    raise ValueError("Angle must be 20..160")
                result = link.send_angle(angle, time.monotonic())
                if result:
                    result.wait_for_publish(timeout=1)
                time.sleep(0.8)
            except ValueError as exc:
                print(exc)
    finally:
        link.close()


if __name__ == "__main__":
    main()
