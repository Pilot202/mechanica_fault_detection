"""
Motor Fault Detection — Backend
================================
Receives vibration features (RMS, Peak Frequency, Max Magnitude) published by
the ESP32 over MQTT, runs them through the trained Random Forest model, and
broadcasts the prediction to any connected dashboard clients over WebSocket.

Matches the ESP32 payload format: "RMS,PeakFrequency,MaxMagnitude"
published to topic: motor/features/vibration
"""

import json
import asyncio
import numpy as np
import joblib
import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from contextlib import asynccontextmanager

# --- Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "motor/features/vibration"
MODEL_PATH = "random_forest_motor_model.pkl"

# --- Load the trained model ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"WARNING: '{MODEL_PATH}' not found. Predictions will be skipped "
          f"until the model file is present in the deployment.")
    model = None

CONDITION_MAP = {
    0: {"status": "Healthy", "color": "green"},
    1: {"status": "Imbalance", "color": "orange"},
    2: {"status": "Loose Mounting", "color": "red"},
}


# --- WebSocket connection manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Dashboard client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Dashboard client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Failed to send to a client, marking for removal: {e}")
                dead_connections.append(connection)
        for dead in dead_connections:
            self.disconnect(dead)


manager = ConnectionManager()

# Reference to FastAPI's event loop, captured at startup, so the MQTT
# background thread can safely hand work back to it.
main_loop: asyncio.AbstractEventLoop | None = None


# --- MQTT callbacks ---
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code {rc}")
    client.subscribe(MQTT_TOPIC)
    print(f"Subscribed to topic: {MQTT_TOPIC}")


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        print(f"Received MQTT message: {payload}")

        features = [float(x) for x in payload.split(',')]

        if len(features) != 3:
            print(f"Unexpected payload shape ({len(features)} values), expected 3. Skipping.")
            return

        if model is None:
            print("Model not loaded — skipping prediction, but data was received correctly.")
            return

        X_live = np.array(features).reshape(1, -1)
        prediction = int(model.predict(X_live)[0])
        probabilities = model.predict_proba(X_live)[0]
        confidence = float(np.max(probabilities)) * 100

        condition_info = CONDITION_MAP.get(prediction, {"status": "Unknown", "color": "grey"})

        result = {
            "rms": features[0],
            "peak_freq": features[1],
            "magnitude": features[2],
            "status": condition_info["status"],
            "color": condition_info["color"],
            "confidence": f"{confidence:.1f}%",
        }
        print(f"Prediction: {result}")

        # Hand the broadcast off to the main FastAPI event loop, since this
        # callback runs on paho-mqtt's own background thread and cannot
        # safely touch WebSocket objects owned by a different event loop.
        if main_loop is not None:
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(json.dumps(result)), main_loop
            )
        else:
            print("Main event loop not ready yet — dropping this message.")

    except Exception as e:
        print(f"Error processing MQTT message: {e}")


# --- App lifespan: start/stop the MQTT client alongside FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop
    main_loop = asyncio.get_running_loop()

    mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()

    print("MQTT client started, connecting in background...")
    yield

    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("MQTT client stopped.")


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keeps the connection alive; dashboard doesn't need to send
            # anything, but we still need to await something so the
            # handler doesn't exit immediately.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- Optional: simple health check, useful for confirming the service is up ---
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "connected_dashboard_clients": len(manager.active_connections),
    }


# --- Optional: simple health check, useful for confirming the service is up ---
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "connected_dashboard_clients": len(manager.active_connections),
    }