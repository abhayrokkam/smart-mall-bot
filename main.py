from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import shutil
import uuid

from modules.utils import push_to_chroma
from modules.engine import MallAssistant
from modules.logging_config import setup_logger

# Root logger
logger = setup_logger(__name__)

# Initialize FastAPI
app = FastAPI(title="Sunway Mall Assistant API", version="1.0")
logger.info("FastAPI app initialized.")

# Initialize Assistant
try:
    assistant = MallAssistant()
    logger.info("MallAssistant initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize MallAssistant.")

# Endpoint: Push data to ChromaDB
@app.post("/push")
async def push_to_chromadb(data_file: UploadFile = File(...)):
    """
    Pushes the uploaded shop data JSON into ChromaDB.
    """
    logger.info(f"Received request to push data from file: {data_file.filename}")
    
    # Save uploaded file temporarily
    temp_dir = "./tmp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.json")
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(data_file.file, f)
    
    try:
        push_to_chroma(file_path)
        logger.info(f"Successfully pushed data from {file_path} to ChromaDB.")
        return {"status": "success", "message": "Data pushed to ChromaDB successfully."}
    except Exception as e:
        logger.error(f"Error pushing data from {file_path} to ChromaDB: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(file_path)

# Request body for /chat
class ChatRequest(BaseModel):
    thread_id: str
    user_query: str

# Endpoint: Chat with the assistant
@app.post("/chat")
def chat_with_bot(request: ChatRequest):
    """
    Processes a user query and returns the chatbot's response.
    """
    logger.info(f"Received chat request for thread_id: {request.thread_id}")
    try:
        config = {"thread_id": request.thread_id}
        result = assistant.process_user_query(request.user_query, config)
        logger.info(f"Successfully processed chat request for thread_id: {request.thread_id}")
        return {
            "response": result["response"],
            "history": [
                {"role": msg.type, "content": msg.content}
                for msg in result["history"]
            ]
        }
    except Exception as e:
        logger.exception(f"Unhandled error processing chat request for thread_id {request.thread_id}: {e}")
        return {"status": "error", "message": str(e)}

# Endpoint: Get chat history
@app.get("/history/{thread_id}")
def get_chat_history(thread_id: str):
    """
    Retrieves full chat history for a given thread_id.
    """
    logger.info(f"Received request for chat history for thread_id: {thread_id}")
    try:
        config = {"thread_id": thread_id}
        result = assistant.graph.get_state(config)
        messages = result.get("messages", [])
        logger.info(f"Successfully retrieved chat history for thread_id: {thread_id}")
        
        return {
            "thread_id": thread_id,
            "history": [
                {"role": msg.type, "content": msg.content}
                for msg in messages
            ]
        }
    except Exception as e:
        logger.error(f"Error retrieving chat history for thread_id {thread_id}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# Run the file direclty
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with Uvicorn.")
    uvicorn.run(app, host="0.0.0.0", port=8000)