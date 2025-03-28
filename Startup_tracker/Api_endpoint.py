from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import openai
from dotenv import load_dotenv
import asyncio
import tempfile
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Create FastAPI app
app = FastAPI(
    title="Document Summarization API",
    description="API for summarizing documents using OpenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for jobs
summary_jobs = {}

class SummaryRequest(BaseModel):
    file_id: str
    summary_type: str = "concise"  # concise, detailed, or bullet_points
    max_length: Optional[int] = 500

class SummaryResponse(BaseModel):
    job_id: str
    status: str
    summary: Optional[str] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Document Summarization API"}

@app.post("/api/upload-document", status_code=200)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for summarization"""
    try:
        # Generate a unique file ID
        file_id = str(uuid.uuid4())
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # Read the uploaded file
            content = await file.read()
            # Write to temp file
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Store the file path with the file_id
        summary_jobs[file_id] = {
            "file_path": temp_path,
            "original_filename": file.filename,
            "status": "uploaded",
            "upload_time": time.time(),
            "summary": None,
            "error": None
        }
        
        return {"file_id": file_id, "filename": file.filename, "status": "uploaded"}
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

async def summarize_document(job_id: str, file_path: str, summary_type: str, max_length: int):
    """Background task to summarize a document"""
    try:
        # Update job status
        summary_jobs[job_id]["status"] = "processing"
        
        # Read the file
        with open(file_path, 'rb') as file:
            file_content = file.read()
        
        # Extract text from file (simplified - you might need a library for different file types)
        # Here we assume it's a text file
        try:
            text = file_content.decode('utf-8')
        except UnicodeDecodeError:
            # If it's not a text file, you'd need to use appropriate libraries
            # like PyPDF2 for PDFs, docx for Word docs, etc.
            raise ValueError("Unsupported file format. Only text files are supported in this example.")
        
        # Create prompt based on summary type
        if summary_type == "concise":
            prompt = f"Provide a concise summary of the following text in {max_length} words or less:\n\n{text}"
        elif summary_type == "detailed":
            prompt = f"Provide a detailed summary of the following text in {max_length} words or less:\n\n{text}"
        elif summary_type == "bullet_points":
            prompt = f"Summarize the following text as a list of bullet points (maximum {max_length} words):\n\n{text}"
        else:
            prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes documents accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        # Extract the summary
        summary = response.choices[0].message.content
        
        # Update job with summary
        summary_jobs[job_id]["summary"] = summary
        summary_jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Error summarizing document: {str(e)}")
        summary_jobs[job_id]["status"] = "failed"
        summary_jobs[job_id]["error"] = str(e)
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error deleting temporary file: {str(e)}")

@app.post("/api/summarize", response_model=SummaryResponse)
async def create_summary(request: SummaryRequest, background_tasks: BackgroundTasks):
    """Request a document summary"""
    file_id = request.file_id
    
    # Check if file exists
    if file_id not in summary_jobs:
        raise HTTPException(status_code=404, detail=f"File ID {file_id} not found")
    
    job = summary_jobs[file_id]
    
    # Generate a job ID
    job_id = str(uuid.uuid4())
    
    # Create a new job entry
    summary_jobs[job_id] = {
        "file_id": file_id,
        "status": "queued",
        "summary_type": request.summary_type,
        "max_length": request.max_length,
        "request_time": time.time(),
        "summary": None,
        "error": None
    }
    
    # Start the background task
    background_tasks.add_task(
        summarize_document,
        job_id,
        job["file_path"],
        request.summary_type,
        request.max_length
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "summary": None,
        "error": None
    }

@app.get("/api/job/{job_id}", response_model=SummaryResponse)
async def get_job_status(job_id: str):
    """Get the status of a summarization job"""
    if job_id not in summary_jobs:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    job = summary_jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "summary": job.get("summary"),
        "error": job.get("error")
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)