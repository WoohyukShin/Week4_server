from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import os

# Create FastAPI app
app = FastAPI(title="Air Traffic Control Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Air Traffic Control Backend is running!", "status": "ready"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "port": os.environ.get("PORT", "Not set")}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("클라이언트 연결됨")
    
    # Send connection confirmation
    await websocket.send_text(json.dumps({
        "type": "connection_established",
        "message": "Connected to Railway backend"
    }))
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            print(f"프론트에서 메시지 수신: {data}")
            
            try:
                message_data = json.loads(data)
                
                if message_data.get("type") == "start_simulation":
                    # For now, just send a test response
                    response = {
                        "type": "start_simulation_response",
                        "success": True,
                        "message": "Test simulation started"
                    }
                    await websocket.send_text(json.dumps(response))
                else:
                    # Echo back the message
                    response = {
                        "type": "echo",
                        "message": f"Received: {message_data}"
                    }
                    await websocket.send_text(json.dumps(response))
                    
            except json.JSONDecodeError:
                print("Invalid JSON received")
                
    except WebSocketDisconnect:
        print("클라이언트 연결 종료") 