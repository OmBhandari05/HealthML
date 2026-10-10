
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="Health ML Risk Assessment API",
    version="0.1.0",
    description=(
        "Educational ML risk-assessment prototype. "
        "It is not a medical diagnosis system."
    ),
)

# Allow requests from both Vite development and preview servers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "name": "Health ML Risk Assessment API",
        "status": "running",
        "supported_models": ["diabetes", "cardiovascular"],
        "disclaimer": (
            "Research/educational prototype only; "
            "not medical diagnosis."
        ),
    }
