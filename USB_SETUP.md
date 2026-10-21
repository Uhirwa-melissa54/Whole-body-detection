# Current setup: user's USB angle-only Arduino sketch

The tracker now matches the supplied Arduino code: 115200 baud, integer angle followed by newline, replies OK:<angle>:<target_steps> and HB:<angle>:<current_steps>/<total_steps>. No MQTT broker is required. MQTT files are preserved separately for future use.

Upload the USB-SERIAL 28BYJ-48 TACHOMETER TRACKER sketch you pasted. That sketch performs a startup sweep, assumes a physically zeroed needle at 0 degrees, and defines 1024 steps as a 180-degree sweep. This step calibration differs from the earlier 4096-per-revolution firmware; check real travel on your motor.

Before powering/resetting or opening the COM port, place the UNPOWERED needle at its 0-degree position and check full sweep clearance. Opening COM9 can reset the board and run its sweep. Do not turn the powered needle by hand. Python waits up to 20 seconds for startup/OK/HB replies. Close Arduino Serial Monitor before Python.

```powershell
cd D:\RoboticsY3\face-recognition-5pt
.\.venv\Scripts\python.exe -m src.hardware_tracking --port COM9
```

Enter 95, 85, or 90, then Q. Only numeric angles are sent; HOME, OFF, STOP and PING are unsupported. The status differentiates accepted target steps (OK) from calculated current steps (HB). Neither is measured motor feedback.

Then run:

```powershell
.\.venv\Scripts\python.exe -m src.face_tracking --any-face --camera 1 --port COM9
```

No enrollment. Default needle limits are 15..165. Q closes USB, SPACE pauses new updates, R commands 90, I reverses direction. H and O have been removed. On face loss Python stops sending new angles. This Arduino sketch has no STOP or watchdog: it may finish the last move after pause, face loss or quit and retains coil power. Physically disconnect power if you need all motion/coils off. Do not interpret closing Python as a hardware emergency stop.

Motor wiring remains D1->IN1, D2->IN2, D5->IN3, D6->IN4 and common ground. A motor rated 5 V needs a suitable 5 V supply. The communication change alone does not prove hardware power/driver faults resolved.
