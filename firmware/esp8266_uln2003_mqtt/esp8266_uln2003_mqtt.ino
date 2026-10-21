/* ESP8266 / 28BYJ-48 / ULN2003 MQTT face tracker.
   D1 -> IN1, D2 -> IN2, D5 -> IN3, D6 -> IN4; shared ground.
   Install AccelStepper, PubSubClient, ArduinoJson 6 or 7.
   Configure Wi-Fi password and broker address below.
   USB is used for uploading/diagnostic logs, not motion commands.
*/
#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <AccelStepper.h>
#include <math.h>
#include <string.h>

// EDIT THESE SETTINGS BEFORE UPLOAD
const char* WIFI_SSID = "Main Hall";
const char* WIFI_PASSWORD = "ENTER_WIFI_PASSWORD";
const char* MQTT_HOST = "10.12.72.168"; // PC Wi-Fi IPv4; recheck if it changes
const uint16_t MQTT_PORT = 1883;
const char* MQTT_USERNAME = ""; // leave empty for supplied local demo config
const char* MQTT_PASSWORD = "";

const char* COMMAND_TOPIC = "face-centering/motor/command";
const char* STATUS_TOPIC = "face-centering/motor/status";
const long STEPS_PER_REV = 4096;
const unsigned long WATCHDOG_MS = 1200;
AccelStepper motor(AccelStepper::HALF4WIRE, 5, 14, 4, 12);
WiFiClient network;
PubSubClient mqtt(network);
bool homed = false;
bool enabled = false;
bool watchdogStopped = true;
unsigned long lastAngle = 0;
unsigned long lastStatus = 0;
unsigned long lastWifiAttempt = 0;
unsigned long lastMqttAttempt = 0;

void haltMotor() {
  motor.setCurrentPosition(motor.currentPosition());
  watchdogStopped = true;
}
void report(const char* status) {
  if (!mqtt.connected()) return;
  StaticJsonDocument<256> doc;
  doc["status"] = status;
  doc["homed"] = homed;
  doc["enabled"] = enabled;
  doc["angle"] = 90.0f + motor.currentPosition() * 360.0f / STEPS_PER_REV;
  doc["position_steps"] = motor.currentPosition();
  doc["target_steps"] = motor.targetPosition();
  char payload[256];
  size_t length = serializeJson(doc, payload, sizeof(payload));
  mqtt.publish(STATUS_TOPIC, (const uint8_t*)payload, length, false);
  Serial.println(payload);
}
void receive(char* topic, byte* payload, unsigned int length) {
  if (strcmp(topic, COMMAND_TOPIC) != 0) return;
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, payload, length)) { report("invalid_json"); return; }
  const char* command = doc["command"] | "";
  if (strcmp(command, "PING") == 0) { report("ready"); return; }
  if (strcmp(command, "STOP") == 0) { haltMotor(); report("stopped"); return; }
  if (strcmp(command, "OFF") == 0) {
    haltMotor(); motor.disableOutputs(); homed = enabled = false;
    report("off_manual_center_then_HOME"); return;
  }
  if (strcmp(command, "HOME") == 0) {
    haltMotor(); motor.setCurrentPosition(0); motor.enableOutputs();
    homed = enabled = true;
    report("homed_manual_center"); return;
  }
  if (!doc["angle"].is<float>() && !doc["angle"].is<int>()) {
    report("missing_angle"); return;
  }
  float angle = doc["angle"].as<float>();
  if (!isfinite(angle) || angle < 15 || angle > 165) { report("invalid_angle_15_to_165"); return; }
  if (!homed) { report("manual_center_then_HOME_required"); return; }
  motor.enableOutputs(); enabled = true;
  motor.moveTo(lroundf((angle - 90.0f) * STEPS_PER_REV / 360.0f));
  lastAngle = millis(); watchdogStopped = false;
}
void setup() {
  Serial.begin(115200);
  motor.setMaxSpeed(500.0); motor.setAcceleration(250.0);
  motor.disableOutputs();
  WiFi.mode(WIFI_STA); WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  lastWifiAttempt = millis();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(receive);
  mqtt.setBufferSize(512);
  mqtt.setSocketTimeout(1);
  Serial.println("MQTT tracker started. OFF -> manually center -> HOME.");
}
void loop() {
  if (!watchdogStopped && (millis() - lastAngle > WATCHDOG_MS || !mqtt.connected())) {
    haltMotor(); report("watchdog_stopped");
  }
  motor.run();
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - lastWifiAttempt >= 10000) {
      lastWifiAttempt = millis(); WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      Serial.println("Retrying Wi-Fi");
    }
  } else if (!mqtt.connected()) {
    // Synchronous connect can pause step generation; stop before retrying.
    haltMotor();
    if (millis() - lastMqttAttempt >= 5000) {
      lastMqttAttempt = millis();
      String clientId = "face-motor-" + String(ESP.getChipId(), HEX);
      const char* user = strlen(MQTT_USERNAME) ? MQTT_USERNAME : nullptr;
      const char* pass = strlen(MQTT_USERNAME) ? MQTT_PASSWORD : nullptr;
      if (mqtt.connect(clientId.c_str(), user, pass, STATUS_TOPIC, 0, false,
                       "{\"status\":\"offline\",\"homed\":false}")) {
        mqtt.subscribe(COMMAND_TOPIC); report("online");
        Serial.print("ESP8266 IP: "); Serial.println(WiFi.localIP());
      } else { Serial.print("MQTT connection failed: "); Serial.println(mqtt.state()); }
    }
  } else {
    mqtt.loop();
  }
  motor.run();
  if (mqtt.connected() && millis() - lastStatus >= 1000) {
    lastStatus = millis();
    report(motor.distanceToGo() ? "moving" : "at_target");
  }
  yield();
}
