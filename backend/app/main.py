from fastapi import FastAPI
from app.routes import router

app = FastAPI()

@app.get("/")
def home():
    return{"message": "Truth Detection API"}

app.include_router(router)