from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info("Root endpoint accessed")
    return {"message": "Air Traffic Control Backend is running!", "status": "ready"}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "port": os.environ.get("PORT", "Not set")}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("클라이언트 연결됨")
    
    # Send connection confirmation
    await websocket.send_text(json.dumps({
        "type": "connection_established",
        "message": "Connected to Render backend"
    }))
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            logger.info(f"프론트에서 메시지 수신: {data}")
            
            try:
                message_data = json.loads(data)
                
                if message_data.get("type") == "start_simulation":
                    # For now, just send a test response
                    response = {
                        "type": "start_simulation_response",
                        "success": True,
                        "message": "Test simulation started on Render"
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
                logger.error("Invalid JSON received")
                
    except WebSocketDisconnect:
        logger.info("클라이언트 연결 종료")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 