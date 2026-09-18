"""
FastAPI application entrypoint for AI Support Ticket Analysis System.
Configures lifespan events, CORS, OpenAPI documentation, and API routing.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.data_layer.loader import initialize_database

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("support_ticket_analysis")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event: loads the dataset and initializes the in-memory SQLite database
    upon application startup.
    """
    logger.info("Initializing support ticket analysis system...")
    initialize_database()
    logger.info("System initialization complete. Ready to receive requests.")
    yield
    logger.info("System shutting down cleanly.")


app = FastAPI(
    title="AI Customer Support Ticket Analysis API",
    description=(
        "Production-ready AI-powered analysis system for customer support tickets. "
        "Provides natural language to SQL translation, explainable statistical & rule-based "
        "anomaly detection, and dataset analytics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    from src.config import settings
    uvicorn.run(
        "main:app",
        host=settings.FASTAPI_HOST,
        port=settings.FASTAPI_PORT,
        reload=False,
    )
