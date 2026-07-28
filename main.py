import json
import numpy as np
import paho.mqtt.client as mqtt
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from contextlib import asynccontextmanager
import asyncio

# --- Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "motor/features/vibration"
MODEL_PATH = "/random_forest_motor_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Warning: {MODEL_PATH} not found. Ensure the model is trained and saved.")
    model = None

CONDITION_MAP = {
    0: {"status": "Healthy", "color": "green"},
    1: {"status": "Imbalance", "color": "orange"},
    2: {"status": "Loose Mounting", "color": "red"}
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Broadcast error: {e}")  # don't swallow silently while debugging

manager = ConnectionManager()

# Will hold a reference to the FastAPI/main thread's event loop
main_loop: asyncio.AbstractEventLoop | None = None

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        print(f"Received MQTT message: {payload}")  # add this to confirm messages arrive at all
        features = [float(x) for x in payload.split(',')]

        if len(features) == 3 and model:
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
                "confidence": f"{confidence:.1f}%"
            }

            # KEY FIX: schedule the coroutine on the MAIN event loop (the one
            # that owns the WebSocket connections), from this MQTT thread.
            if main_loop is not None:
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast(json.dumps(result)), main_loop
                )
            else:
                print("Main event loop not ready yet, dropping message")

    except Exception as e:
        print(f"Error processing message: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop
    main_loop = asyncio.get_running_loop()  # capture FastAPI's loop while we're on it

    mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    yield
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
