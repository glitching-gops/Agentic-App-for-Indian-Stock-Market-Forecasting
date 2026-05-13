TICKERS = {
    "M&M.NS": {"company": "Mahindra & Mahindra", "sector": "Automobile"},
    "TVSMOTOR.NS": {"company": "TVS Motor Company", "sector": "Automobile"},
    "BAJAJ-AUTO.NS": {"company": "Bajaj Auto", "sector": "Automobile"},
    "EICHERMOT.NS": {"company": "Eicher Motors", "sector": "Automobile"},
    "BOSCHLTD.NS": {"company": "Bosch", "sector": "Automobile"},
    "HDFCBANK.NS": {"company": "HDFC Bank", "sector": "Banking & Finance"},
    "PNBHOUSING.NS": {"company": "PNB Housing Finance", "sector": "Banking & Finance"},
    "BAJFINANCE.NS": {"company": "Bajaj Finance", "sector": "Banking & Finance"},
    "ICICIPRULI.NS": {"company": "ICICI Prudential Life", "sector": "Banking & Finance"},
    "BAJAJFINSV.NS": {"company": "Bajaj Finserv", "sector": "Banking & Finance"},
    "TRENT.NS": {"company": "Trent", "sector": "Consumer Discretionary"},
    "BERGEPAINT.NS": {"company": "Berger Paints", "sector": "Consumer Discretionary"},
    "PIDILITIND.NS": {"company": "Pidilite Industries", "sector": "Consumer Discretionary"},
    "ASIANPAINT.NS": {"company": "Asian Paints", "sector": "Consumer Discretionary"},
    "DMART.NS": {"company": "Avenue Supermarts (DMart)", "sector": "Consumer Discretionary"},
    "RELIANCE.NS": {"company": "Reliance Industries", "sector": "Energy"},
    "ADANIENT.NS": {"company": "Adani Enterprises", "sector": "Energy"},
    "GAIL.NS": {"company": "GAIL India", "sector": "Energy"},
    "ADANIPORTS.NS": {"company": "Adani Ports & SEZ", "sector": "Energy"},
    "NTPC.NS": {"company": "NTPC", "sector": "Energy"},
    "COLPAL.NS": {"company": "Colgate-Palmolive India", "sector": "FMCG"},
    "EMAMILTD.NS": {"company": "Emami", "sector": "FMCG"},
    "VBL.NS": {"company": "Varun Beverages", "sector": "FMCG"},
    "GODREJCP.NS": {"company": "Godrej Consumer Products", "sector": "FMCG"},
    "DABUR.NS": {"company": "Dabur India", "sector": "FMCG"},
    "MPHASIS.NS": {"company": "Mphasis", "sector": "Information Technology"},
    "WIPRO.NS": {"company": "Wipro", "sector": "Information Technology"},
    "PERSISTENT.NS": {"company": "Persistent Systems", "sector": "Information Technology"},
    "HCLTECH.NS": {"company": "HCL Technologies", "sector": "Information Technology"},
    "TECHM.NS": {"company": "Tech Mahindra", "sector": "Information Technology"},
    "HAVELLS.NS": {"company": "Havells India", "sector": "Infrastructure"},
    "SUPREMEIND.NS": {"company": "Supreme Industries", "sector": "Infrastructure"},
    "ADANIGREEN.NS": {"company": "Adani Green Energy", "sector": "Infrastructure"},
    "SIEMENS.NS": {"company": "Siemens India", "sector": "Infrastructure"},
    "CUMMINSIND.NS": {"company": "Cummins India", "sector": "Infrastructure"},
    "JSWSTEEL.NS": {"company": "JSW Steel", "sector": "Metals & Mining"},
    "VEDL.NS": {"company": "Vedanta", "sector": "Metals & Mining"},
    "NMDC.NS": {"company": "NMDC", "sector": "Metals & Mining"},
    "JINDALSTEL.NS": {"company": "Jindal Steel & Power", "sector": "Metals & Mining"},
    "SAIL.NS": {"company": "Steel Authority of India", "sector": "Metals & Mining"},
    "CIPLA.NS": {"company": "Cipla", "sector": "Pharmaceuticals"},
    "SUNPHARMA.NS": {"company": "Sun Pharmaceutical", "sector": "Pharmaceuticals"},
    "ALKEM.NS": {"company": "Alkem Laboratories", "sector": "Pharmaceuticals"},
    "DRREDDY.NS": {"company": "Dr. Reddy's Laboratories", "sector": "Pharmaceuticals"},
    "ABBOTINDIA.NS": {"company": "Abbott India", "sector": "Pharmaceuticals"},
    "OBEROIRLTY.NS": {"company": "Oberoi Realty", "sector": "Real Estate"},
    "GODREJPROP.NS": {"company": "Godrej Properties", "sector": "Real Estate"},
    "DLF.NS": {"company": "DLF", "sector": "Real Estate"},
    "PRESTIGE.NS": {"company": "Prestige Estates", "sector": "Real Estate"},
    "PHOENIXLTD.NS": {"company": "Phoenix Mills", "sector": "Real Estate"},
    "BHARTIARTL.NS": {"company": "Bharti Airtel", "sector": "Telecom"},
    "IDEA.NS": {"company": "Vodafone Idea", "sector": "Telecom"},
    "INDUSTOWER.NS": {"company": "Indus Towers", "sector": "Telecom"},
}

SECTOR_INDICES = {
    "Banking & Finance":      "^NSEBANK",
    "Information Technology": "^CNXIT",
    "Energy":                 "^CNXENERGY",
    "FMCG":                   "^CNXFMCG",
    "Automobile":             "^CNXAUTO",
    "Pharmaceuticals":        "^CNXPHARMA",
    "Metals & Mining":        "^CNXMETAL",
    "Infrastructure":         "^CNXINFRA",
    "Telecom":                "^CNXMEDIA",
    "Real Estate":            "^CNXREALTY",
    "Consumer Discretionary": "^CNXCONSUM",
}

def get_all_sectors() -> list[str]:
    """Returns a sorted list of all unique sectors in the current universe."""
    return sorted(set(v["sector"] for v in TICKERS.values()))

def get_tickers_by_sector(sector: str) -> list[str]:
    """Returns all tickers in a given sector."""
    return [t for t, v in TICKERS.items() if v["sector"] == sector]

def get_company(ticker: str) -> str:
    return TICKERS.get(ticker, {}).get("company", ticker)

def get_sector(ticker: str) -> str:
    return TICKERS.get(ticker, {}).get("sector", "Unknown")
