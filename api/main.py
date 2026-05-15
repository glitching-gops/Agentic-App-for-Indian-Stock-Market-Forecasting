"""
FastAPI application entry point.
Initialises the database, mounts routers, configures CORS,
and starts the APScheduler daily pipeline job on startup.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from data.db import init_db
from api.routers import stocks, forecasts, leaderboard, admin, signals, sentiment
from scheduler import start_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run on startup (DB init is handled by render_start.sh)
    try:
        start_scheduler()
    except Exception as e:
        print(f"Warning: Scheduler failed to start: {e}")
    yield
    # Run on shutdown (if needed)

app = FastAPI(
    title="Indian Stock Market Forecasting API",
    description="Agentic multi-agent system for NSE stock price forecasting",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stocks.router,     prefix="/api/stocks",     tags=["Stocks"])
app.include_router(forecasts.router,  prefix="/api/forecasts",  tags=["Forecasts"])
app.include_router(leaderboard.router,prefix="/api/leaderboard",tags=["Leaderboard"])
app.include_router(admin.router,      prefix="/api/admin",      tags=["Admin"])
app.include_router(signals.router,    prefix="/api/signals",    tags=["Signals"])
app.include_router(sentiment.router,  prefix="/api/sentiment",  tags=["Sentiment"])

@app.get("/")
def read_root():
    return {"message": "ZeRO Stock Forecast API is running", "docs": "/docs", "health": "/api/health"}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
