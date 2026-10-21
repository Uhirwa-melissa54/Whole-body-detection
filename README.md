# Current setup: USB angle-only sketch

Read [USB_SETUP.md](USB_SETUP.md). This supersedes older MQTT and HOME/OFF instructions below.

## Whole-person face association, camera only

Enroll with `python -m src.enroll --camera 1`; camera 1 is now the default. Install YOLO once with `python -m pip install -r requirements-yolo.txt`, then run `python -m src.whole_person_tracking --target Alice --camera 0 --mirror` to evaluate an enrolled person without hardware. Its default is the CPU-only Ultralytics YOLO11 nano model (`yolo11n.pt`) restricted to COCO's person class. The model weights download on first use. Use `--detector hog` to use the original OpenCV HOG fallback without YOLO. `--target` uses your ArcFace enrollment database to acquire and confirm only that person's visible face and associated body. `--any-face` remains available for generic evaluation. When the associated target face is visible it reports `CONFIRMED` at confidence 1.00. When only the associated body remains it reports `TEMPORARY`, and confidence follows exponential decay; after the temporary period or body loss it reports `UNKNOWN` at 0.00. This explicitly distinguishes an observed face from a spatially continued body target. The module requests MJPEG, drops stale camera frames, and runs person detection every three frames by default to improve responsiveness; set `--body-every 1` for every-frame body detection.

Use head turns and brief occlusions to test Temporary, then step away or cover both face and body to test Unknown. Cross paths with another person to evaluate whether the HOG body association stays with the target. `R` clears the selected target; `F` toggles full screen. The body-centering angle is a software evaluation signal and is not sent to hardware.

# Current setup: MQTT

Use [MQTT_SETUP.md](MQTT_SETUP.md) for the current Python and ESP8266 setup. Earlier USB instructions below are historical.

# Face recognition with ESP8266 USB stepper tracking

This integrates FaceLocking's horizontal motor control into your existing five-point ArcFace identity tracker. Recognition, face locking, smile detection, blink counting, and vertical/horizontal screen signals remain in `src.face_tracking`. USB replaces the reference project's MQTT broker. No Wi-Fi credentials or broker are needed.

## Hardware and power first

Confirmed wiring: D1 (GPIO5) -> ULN2003 IN1; D2 (GPIO4) -> IN2; D5 (GPIO14) -> IN3; D6 (GPIO12) -> IN4. Plug the motor into the ULN2003 motor socket. Connect ESP8266 GND, ULN2003 minus, and motor supply minus together. Connect USB from PC to ESP8266.

Check the voltage printed on the motor. For a 5 V 28BYJ-48, disconnect ULN2003 plus from ESP8266 3V3 and connect ULN2003 plus to a regulated 5 V motor supply with sufficient current. A separate 5 V supply rated at least 1 A is a practical starting point for one motor. Keep motor power separate from the board's USB 5 V rail; share ground only. For a motor marked 12 V, use its rated supply for the driver; do not feed that voltage into the ESP8266. Verify driver compatibility before powering it.

The firmware assumes a 28BYJ-48 four-coil unipolar motor driven by ULN2003, not a tachometer pulse sensor. The needle angle indicates the face's position; there is no RPM measurement. One motor controls horizontal motion only. Vertical direction is displayed on screen.

## Flash the ESP8266

1. Install Arduino IDE. Add `https://arduino.esp8266.com/stable/package_esp8266com_index.json` under Preferences -> Additional Boards Manager URLs. Install the ESP8266 platform using Boards Manager.
2. Select your actual board, for example NodeMCU 1.0 (ESP-12E Module) or LOLIN(WEMOS) D1 mini, and its COM port.
3. Install **AccelStepper** by Mike McCauley from Library Manager.
4. Open `firmware/esp8266_uln2003_usb/esp8266_uln2003_usb.ino` and upload it. Close Serial Monitor before Python opens the same port.

There is no automatic startup sweep. Motor outputs remain disabled until HOME. A stepper has no built-in position sensor: HOME records the position you manually set; it does not detect a stop. Nominal half-step calibration is 4096 steps per revolution. Adjust `STEPS_PER_REV` for your motor/gearbox. The driver telemetry reports calculated step position, not a measured physical position.

## Python installation

Open PowerShell:

```powershell
Set-Location D:\RoboticsY3\face-recognition-5pt
.\.venv\Scripts\python.exe -m pip install -r requirements-hardware.txt
.\.venv\Scripts\python.exe -m src.hardware_tracking --list-ports
```

Your existing environment and ArcFace model are reused. The original `requirements.txt` describes your existing recognition environment; USB support is separate. If recognition is not installed already, install the original requirements too. Replace COM5 in the examples with the listed ESP8266 port. If no port appears, use a USB data cable and the USB-UART driver appropriate to your board (often CH340 or CP210x).

## Check the motor without the camera

```powershell
.\.venv\Scripts\python.exe -m src.hardware_tracking --port COM5
```

Enter `OFF`. With coils unpowered, carefully put the needle at the middle of its permitted travel. Enter `HOME` to define this as 90 degrees. Start with `95`, then `85`, then `90`. Enter `PING` for status. Enter `q` to stop and release the COM port. If movement is reversed, use the tracker's `--invert`; do not randomly change coil order. If it vibrates without moving, check rated motor power, common ground, motor connector and IN1..IN4 wiring.

Closing this test and reopening the port can reset the ESP8266. Repeat manual centering/HOME in the tracking window. There is deliberately no `--auto-home` assumption.

## Run recognition and hardware together

```powershell
.\.venv\Scripts\python.exe -m src.face_tracking --target ikirezi --camera 0 --port COM5 --mode pointer
```

Use the exact enrolled name. Current metadata lists Cyuzuzo, Melissa, Noah, and ikirezi; the tracker validates the actual NPZ database. To enroll another person run `python -m src.enroll`, enter the name when prompted, and follow its capture/save instructions. Camera 0 is the first camera; try 1 if necessary.

Before movement: press SPACE to pause, press O to switch coils off, manually center the needle, press H to declare that position 90 degrees, then press SPACE to resume. HOME is required after every board reset and after OFF. The window must have keyboard focus.

- Q: quit; STOP is sent and the COM port is closed.
- SPACE: pause/resume; pause immediately stops the pending trajectory.
- O: release motor coils and pause; requires HOME before further movement.
- H: manual zero at 90 degrees, only after physical centering.
- R: command center (90 degrees); does not physically calibrate the motor.
- I: reverse left/right.
- M: switch fixed-camera pointer and mounted-camera pan modes.

Fixed camera / needle display is the default: left maps toward 15 degrees, center to 90, and right toward 165, with smoothing and a central dead zone. The angle is clipped at the configured limits. Keep the camera stationary for this mode. Example for reversed direction:

```powershell
.\.venv\Scripts\python.exe -m src.face_tracking --target ikirezi --port COM5 --mode pointer --invert
```

Use pan mode only if the camera is physically mounted on the motor's rotating platform:

```powershell
.\.venv\Scripts\python.exe -m src.face_tracking --target ikirezi --port COM5 --mode pan --speed 20 --gain 0.6
```

Pan mode adjusts position over time to bring the face toward the image center. If turning makes the face move farther away from image center, invert direction. Start slowly and confirm the mechanism can carry the camera. Travel limits default to 15..165; narrow them with `--min-angle 60 --max-angle 120` during commissioning. Firmware independently enforces 15..165. Adjust firmware limits only after checking physical clearance. `--smoothing 0.3` smooths more; `--smoothing 0` disables angle filtering.

## Data flow and loss behavior

PC webcam -> Haar/MediaPipe five landmarks -> ArcFace matches enrolled identity -> existing identity lock -> nose/eye horizontal anchor -> angle controller -> newline USB command -> ESP8266 -> AccelStepper -> ULN2003 -> motor/needle. ESP8266 sends step/angle status back over the same USB connection and Python displays it.

Recognition runs on your PC, not on the ESP8266. Hardware never follows a stale box during the existing tracker's visual coasting period: missing detections send STOP and hold the current calculated position. Pause and quit also send STOP. If angle commands stop arriving for 1.2 seconds, firmware stops the trajectory. No automatic return to center on loss or quit; use R deliberately. STOP retains coil holding torque, while OFF releases coils and invalidates calibration. If a manually moved or stalled motor loses steps, OFF -> manually center -> HOME again.

Identity verification retains your existing periodic checks and thresholds. Like the existing tracker, spatial association can temporarily follow a nearby different face before re-verification rejects it. This is a tracking demo rather than access control.

## Troubleshooting and practical limits

- Access denied opening COM: close Arduino Serial Monitor, the hardware test, and other serial applications.
- STATUS homed=0: manually center, then H/HOME. Reconnecting USB can reset the board.
- No recognized target: verify exact enrollment name, lighting, camera index, and enrolled sample quality.
- Jitter or overshoot: reduce gain/speed, increase smoothing, narrow travel, and check mechanical load.
- Disconnect or camera failure: Python cleans up the port and firmware's watchdog stops stale movement.
- Motor buzzing/resetting: verify motor-rated supply, current capacity, and common ground.

The original FaceLocking firmware is available in the sibling project for reference. This integration deliberately avoids its blocking Wi-Fi reconnect loops, assumed startup zero, and misleading pixel-to-step mapping. USB firmware must be flashed before using this project's hardware commands.

Hardware compilation, camera behavior and actual motor movement must be verified on your connected board; software checks alone cannot confirm wiring or physical clearance.

## Current setup: built-in camera 0 and ESP8266 COM9, without enrollment

Your ESP8266 appears as `COM9: Silicon Labs CP210x USB to UART Bridge`. COM4 is the generic communications port. Use COM9.

```powershell
cd D:\RoboticsY3\face-recognition-5pt
.\.venv\Scripts\python.exe -m src.face_tracking --any-face --camera 0 --port COM9 --mode pointer
```

The shorter command `python -m src.face_tracking --any-face` defaults to camera 0, COM9, and pointer mode. This mode does not load ArcFace or an enrollment database. It acquires the largest detected face, then follows it by spatial association. If multiple people appear, it cannot identify a particular person and can switch faces. Use one person in view for motor testing.

With motor-rated power connected and USB firmware uploaded, press O, manually center the unpowered needle, press H, then SPACE to resume. Left/right face movement drives the needle; Q stops. Keep the camera stationary in pointer mode. Use I if the needle moves in the opposite direction. Enrollment is unnecessary for this test.

## Black camera or absent board

Run `python -m src.face_tracking --any-face --camera-only` to isolate the camera. If Windows Camera is also black, open the physical camera shutter, enable the camera using the laptop's camera function key, and check Settings -> Privacy & security -> Camera (Camera access and Let desktop apps access your camera). Close Windows Camera before running Python. The tracker warns when frames contain only zero pixels.

Run `python -m src.hardware_tracking --list-ports` again after reconnecting USB. COM9 is usable only while the CP210x device is present. If it disappears, check the data cable, USB socket and board power; do not select generic COM4 as a substitute. Disconnect motor power while diagnosing the board, especially if the motor was powered from 3V3.

## External camera update

The default camera is now index 1 for an external USB camera. Connect it first, close other camera applications, then test with `python -m src.face_tracking --any-face --camera-only --camera 1`. Once the live image works and the CP210x COM9 port is present, run `python -m src.face_tracking --any-face --camera 1 --port COM9`. Camera indices depend on Windows device enumeration; override with `--camera 2` if the external camera is not index 1. Earlier camera-0 examples refer to the built-in camera.
