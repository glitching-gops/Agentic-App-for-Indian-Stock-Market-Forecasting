from pydantic import BaseModel

class StockInfo(BaseModel):
    ticker: str
    company: str
    sector: str

class StockList(BaseModel):
    stocks: list[StockInfo]
    total: int
