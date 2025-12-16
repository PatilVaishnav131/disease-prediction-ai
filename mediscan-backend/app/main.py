from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

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

@app.post("/predict/brain")
async def brain(file: UploadFile = File(...)):
    return predict_brain_tumor(file)

@app.post("/predict/malaria")
async def malaria(file: UploadFile = File(...)):
    return predict_malaria(file)

@app.post("/predict/skin")
async def skin(file: UploadFile = File(...)):
    return predict_skin(file)
