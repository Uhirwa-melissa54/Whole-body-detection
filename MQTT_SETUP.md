# MQTT setup (current transport)

Python and ESP8266 now use MQTT JSON. This replaces the earlier USB motor commands; USB remains for board power/upload/logs. Camera defaults to 1; no enrollment is required with --any-face. MQTT does not fix a disconnected or unpowered motor.

## Values to fill in
Open firmware/esp8266_uln2003_mqtt/esp8266_uln2003_mqtt.ino:
- WIFI_SSID is Main Hall.
- WIFI_PASSWORD: enter your Wi-Fi password locally.
- MQTT_HOST: the IPv4 address of the PC running Mosquitto. Currently 10.12.72.168; check ipconfig each time the network changes.
- MQTT_PORT: 1883.
- MQTT_USERNAME and MQTT_PASSWORD: blank with the supplied anonymous local-demo broker config; fill them if your broker requires authentication.
The ESP8266 requires a compatible 2.4 GHz Wi-Fi network. This sketch uses a normal SSID/password connection; enterprise login/captive-portal networks may require a suitable separate network. PC and ESP8266 must be able to communicate on the network; shared Wi-Fi does not guarantee this if client isolation is enabled.

## Start a broker on this PC
Install Mosquitto for Windows from https://mosquitto.org/download/ if not already installed. In a separate PowerShell window, run:

```powershell
cd D:\RoboticsY3\face-recognition-5pt
& 'C:\Program Files\mosquitto\mosquitto.exe' -c '.\mqtt\mosquitto.conf' -v
```

Keep this terminal open. If port 1883 is already occupied by a broker service, use/configure that broker or stop the conflicting service before starting this one. The included config listens on the network with anonymous access for a local lab demo; use it only on a trusted network and do not expose it to the internet. Prefer a private hotspot if Main Hall is a shared network. If Windows Firewall blocks access, allow Mosquitto's TCP 1883 listener for the network you intentionally use; do not disable the firewall.

## Upload Arduino firmware
Install ESP8266 board support, AccelStepper, PubSubClient, and ArduinoJson. ArduinoJson 6 or 7 can use this sketch's StaticJsonDocument interface. Fill the settings above, select your actual ESP8266 board and COM9, and upload the complete MQTT .ino file. Do not upload the older USB firmware. At 115200 baud, Serial Monitor should show Wi-Fi/broker connection progress. You can keep it open for logs during MQTT tracking because Python does not open COM9.

Wiring remains D1->IN1, D2->IN2, D5->IN3, D6->IN4, ESP8266 GND->ULN2003 minus, motor plug inserted. Supply motor-rated 5 V to ULN2003 plus and share supply ground. Motor position is open-loop, not a measured sensor reading.

## Test MQTT and motor before camera

```powershell
cd D:\RoboticsY3\face-recognition-5pt
.\.venv\Scripts\python.exe -m pip install -r requirements-hardware.txt
.\.venv\Scripts\python.exe -m src.mqtt_tracking --broker 127.0.0.1
```

Expect live board status. Enter OFF, physically center the needle, HOME, PING, 95, and PING. Live status should show homed=1 and angle near 95. If no ESP8266 telemetry arrives, broker connectivity alone is insufficient: check Wi-Fi credentials, MQTT_HOST, broker listener/firewall and network client isolation. Enter Q to stop before starting camera tracking. Motor angle commands expire after 1.2 seconds; manual tests can stop short on larger travel. The camera tracker sends an angle heartbeat every 0.3 seconds.

## Run face tracking

```powershell
.\.venv\Scripts\python.exe -m src.face_tracking --any-face --camera 1 --broker 127.0.0.1 --mode pointer
```

Python uses localhost because the broker is on the same PC; ESP8266 uses the PC's network IP. For a broker on another machine, use that machine's IP in both places. For broker authentication set FACE_MQTT_USERNAME and FACE_MQTT_PASSWORD in Python's environment; enter the same account in firmware settings.

Press O -> manually center -> H -> SPACE. Q stops. I inverts direction. M switches pointer/pan. Use pointer with a stationary camera; pan only with a motor-mounted camera. --camera-only bypasses MQTT for image testing. Missing face detections stop motion; firmware watchdog stops stale commands/disconnections. No automatic startup movement or unverified homing. Use one person in view in --any-face mode.

Topics: face-centering/motor/command and face-centering/motor/status.
Angle command: {"angle":95}; control command: {"command":"HOME"} (also OFF, STOP, PING).
Commands are never retained. Firmware publishes live status every second. Remove any old retained messages on the command topic before using an existing broker, so it does not replay old motion or HOME commands on reconnect.

Firmware compilation and real Wi-Fi/motor movement still require testing on your board. The local broker is not installed or running automatically by these file changes.
