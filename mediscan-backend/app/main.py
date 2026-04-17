from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.services.brain_service import predict_brain_tumor
from app.services.malaria_service import predict_malaria
from app.services.skin_service import predict_skin

app = FastAPI(title="MediScan AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "MediScan Backend Running"}

# Helper function to validate image content type
def is_valid_image_content_type(content_type: str) -> bool:
    return content_type in ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"]

@app.post("/predict/brain")
async def brain(file: UploadFile = File(...)):
    # Validate content type
    if not is_valid_image_content_type(file.content_type):
        return JSONResponse(
            {"error": "Invalid file type. Please upload an image (JPEG, PNG, WEBP, BMP)."},
            status_code=400
        )
    try:
        contents = await file.read()
        if len(contents) == 0:
            return JSONResponse({"error": "Empty file provided."}, status_code=400)
        # Note: predict_brain_tumor may need to be updated to accept bytes instead of UploadFile
        result = predict_brain_tumor(contents)  # assuming it now accepts bytes
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": f"Error processing brain image: {str(e)}"}, status_code=500)

@app.post("/predict/malaria")
async def malaria(file: UploadFile = File(...)):
    if not is_valid_image_content_type(file.content_type):
        return JSONResponse(
            {"error": "Invalid file type. Please upload an image (JPEG, PNG, WEBP, BMP)."},
            status_code=400
        )
    try:
        contents = await file.read()
        if len(contents) == 0:
            return JSONResponse({"error": "Empty file provided."}, status_code=400)
        result = predict_malaria(contents)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": f"Error processing malaria image: {str(e)}"}, status_code=500)

@app.post("/predict/skin")
async def skin(
    file: UploadFile = File(None),
    url: str = Form(None)
):
    """
    Accept either an image file (upload) or a URL (e.g., from Google Images).
    """
    # Validate input
    if file and url:
        return JSONResponse({"error": "Provide either file or URL, not both"}, status_code=400)
    
    if not file and not url:
        return JSONResponse({"error": "No image provided"}, status_code=400)
    
    try:
        if file:
            # Validate content type for uploaded file
            if not is_valid_image_content_type(file.content_type):
                return JSONResponse(
                    {"error": "Invalid file type. Please upload an image (JPEG, PNG, WEBP, BMP)."},
                    status_code=400
                )
            # Read the uploaded file as bytes
            contents = await file.read()
            if len(contents) == 0:
                return JSONResponse({"error": "Empty file provided."}, status_code=400)
            result = predict_skin(contents)            # bytes → treat as file
        else:  # url provided
            # Basic URL validation (optional)
            if not url.startswith(('http://', 'https://')):
                return JSONResponse({"error": "Invalid URL format"}, status_code=400)
            result = predict_skin(url, from_url=True)  # string → treat as URL
        
        return JSONResponse(result)
    
    except Exception as e:
        # Catch any unexpected errors from the service
        return JSONResponse(
            {"error": f"Prediction failed: {str(e)}"},
            status_code=500
        )