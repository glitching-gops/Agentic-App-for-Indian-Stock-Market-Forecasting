TICKERS = {
    # ── Banking & Finance (existing + new) ──────────────────────────
    "HDFCBANK.NS":      {"company": "HDFC Bank",                    "sector": "Banking & Finance"},
    "ICICIBANK.NS":     {"company": "ICICI Bank",                   "sector": "Banking & Finance"},
    "KOTAKBANK.NS":     {"company": "Kotak Mahindra Bank",          "sector": "Banking & Finance"},
    "SBIN.NS":          {"company": "State Bank of India",          "sector": "Banking & Finance"},
    "BAJFINANCE.NS":    {"company": "Bajaj Finance",                "sector": "Banking & Finance"},
    "AXISBANK.NS":      {"company": "Axis Bank",                    "sector": "Banking & Finance"},
    "INDUSINDBK.NS":    {"company": "IndusInd Bank",                "sector": "Banking & Finance"},
    "BAJAJFINSV.NS":    {"company": "Bajaj Finserv",                "sector": "Banking & Finance"},
    "HDFCLIFE.NS":      {"company": "HDFC Life Insurance",          "sector": "Banking & Finance"},
    "SBILIFE.NS":       {"company": "SBI Life Insurance",           "sector": "Banking & Finance"},
    "ICICIPRULI.NS":    {"company": "ICICI Prudential Life",        "sector": "Banking & Finance"},
    "ICICIGI.NS":       {"company": "ICICI Lombard General Ins.",   "sector": "Banking & Finance"},
    "PNBHOUSING.NS":    {"company": "PNB Housing Finance",          "sector": "Banking & Finance"},
    "MUTHOOTFIN.NS":    {"company": "Muthoot Finance",              "sector": "Banking & Finance"},
    "PNB.NS":           {"company": "Punjab National Bank",         "sector": "Banking & Finance"},
    "BANKBARODA.NS":    {"company": "Bank of Baroda",               "sector": "Banking & Finance"},
    "SHRIRAMFIN.NS":    {"company": "Shriram Finance",              "sector": "Banking & Finance"},
    "CHOLAFIN.NS":      {"company": "Cholamandalam Investment",     "sector": "Banking & Finance"},
    "CANBK.NS":         {"company": "Canara Bank",                  "sector": "Banking & Finance"},

    # ── Information Technology (existing + new) ──────────────────────
    "TCS.NS":           {"company": "Tata Consultancy Services",    "sector": "Information Technology"},
    "INFY.NS":          {"company": "Infosys",                      "sector": "Information Technology"},
    "WIPRO.NS":         {"company": "Wipro",                        "sector": "Information Technology"},
    "HCLTECH.NS":       {"company": "HCL Technologies",             "sector": "Information Technology"},
    "TECHM.NS":         {"company": "Tech Mahindra",                "sector": "Information Technology"},
    "MPHASIS.NS":       {"company": "Mphasis",                      "sector": "Information Technology"},
    "PERSISTENT.NS":    {"company": "Persistent Systems",           "sector": "Information Technology"},
    "COFORGE.NS":       {"company": "Coforge",                      "sector": "Information Technology"},
    "OFSS.NS":          {"company": "Oracle Financial Services",    "sector": "Information Technology"},
    "DIXON.NS":         {"company": "Dixon Technologies",           "sector": "Information Technology"},

    # ── Energy & Oil (existing + new) ───────────────────────────────
    "RELIANCE.NS":      {"company": "Reliance Industries",          "sector": "Energy"},
    "ONGC.NS":          {"company": "ONGC",                         "sector": "Energy"},
    "BPCL.NS":          {"company": "BPCL",                         "sector": "Energy"},
    "IOC.NS":           {"company": "Indian Oil Corporation",       "sector": "Energy"},
    "GAIL.NS":          {"company": "GAIL India",                   "sector": "Energy"},
    "COALINDIA.NS":     {"company": "Coal India",                   "sector": "Energy"},
    "NTPC.NS":          {"company": "NTPC",                         "sector": "Energy"},
    "ADANIPORTS.NS":    {"company": "Adani Ports & SEZ",            "sector": "Energy"},
    "ADANIENT.NS":      {"company": "Adani Enterprises",            "sector": "Energy"},

    # ── FMCG (existing + new) ────────────────────────────────────────
    "ITC.NS":           {"company": "ITC",                          "sector": "FMCG"},
    "HINDUNILVR.NS":    {"company": "Hindustan Unilever",           "sector": "FMCG"},
    "NESTLEIND.NS":     {"company": "Nestle India",                 "sector": "FMCG"},
    "BRITANNIA.NS":     {"company": "Britannia Industries",         "sector": "FMCG"},
    "DABUR.NS":         {"company": "Dabur India",                  "sector": "FMCG"},
    "MARICO.NS":        {"company": "Marico",                       "sector": "FMCG"},
    "GODREJCP.NS":      {"company": "Godrej Consumer Products",     "sector": "FMCG"},
    "COLPAL.NS":        {"company": "Colgate-Palmolive India",      "sector": "FMCG"},
    "EMAMILTD.NS":      {"company": "Emami",                        "sector": "FMCG"},
    "VBL.NS":           {"company": "Varun Beverages",              "sector": "FMCG"},

    # ── Automobile (existing + new) ──────────────────────────────────
    "MARUTI.NS":        {"company": "Maruti Suzuki",                "sector": "Automobile"},
    "TATAMOTORS.NS":    {"company": "Tata Motors",                  "sector": "Automobile"},
    "M&M.NS":           {"company": "Mahindra & Mahindra",          "sector": "Automobile"},
    "BAJAJ-AUTO.NS":    {"company": "Bajaj Auto",                   "sector": "Automobile"},
    "HEROMOTOCO.NS":    {"company": "Hero MotoCorp",                "sector": "Automobile"},
    "EICHERMOT.NS":     {"company": "Eicher Motors",                "sector": "Automobile"},
    "TVSMOTOR.NS":      {"company": "TVS Motor Company",            "sector": "Automobile"},
    "ASHOKLEY.NS":      {"company": "Ashok Leyland",                "sector": "Automobile"},
    "BOSCHLTD.NS":      {"company": "Bosch",                        "sector": "Automobile"},
    "MOTHERSON.NS":     {"company": "Samvardhana Motherson",        "sector": "Automobile"},

    # ── Pharmaceuticals ─────────────────────────────────────────────
    "SUNPHARMA.NS":     {"company": "Sun Pharmaceutical",           "sector": "Pharmaceuticals"},
    "DRREDDY.NS":       {"company": "Dr. Reddy's Laboratories",     "sector": "Pharmaceuticals"},
    "CIPLA.NS":         {"company": "Cipla",                        "sector": "Pharmaceuticals"},
    "DIVISLAB.NS":      {"company": "Divi's Laboratories",          "sector": "Pharmaceuticals"},
    "BIOCON.NS":        {"company": "Biocon",                       "sector": "Pharmaceuticals"},
    "AUROPHARMA.NS":    {"company": "Aurobindo Pharma",             "sector": "Pharmaceuticals"},
    "LUPIN.NS":         {"company": "Lupin",                        "sector": "Pharmaceuticals"},
    "TORNTPHARM.NS":    {"company": "Torrent Pharmaceuticals",      "sector": "Pharmaceuticals"},
    "ALKEM.NS":         {"company": "Alkem Laboratories",           "sector": "Pharmaceuticals"},
    "ABBOTINDIA.NS":    {"company": "Abbott India",                 "sector": "Pharmaceuticals"},
    "ZYDUSLIFE.NS":     {"company": "Zydus Lifesciences",           "sector": "Pharmaceuticals"},

    # ── Metals & Mining ─────────────────────────────────────────────
    "TATASTEEL.NS":     {"company": "Tata Steel",                   "sector": "Metals & Mining"},
    "JSWSTEEL.NS":      {"company": "JSW Steel",                    "sector": "Metals & Mining"},
    "HINDALCO.NS":      {"company": "Hindalco Industries",          "sector": "Metals & Mining"},
    "VEDL.NS":          {"company": "Vedanta",                      "sector": "Metals & Mining"},
    "SAIL.NS":          {"company": "Steel Authority of India",     "sector": "Metals & Mining"},
    "NMDC.NS":          {"company": "NMDC",                         "sector": "Metals & Mining"},
    "JINDALSTEL.NS":    {"company": "Jindal Steel & Power",         "sector": "Metals & Mining"},
    "APLAPOLLO.NS":     {"company": "APL Apollo Tubes",             "sector": "Metals & Mining"},

    # ── Infrastructure & Industrial ──────────────────────────────────
    "LT.NS":            {"company": "Larsen & Toubro",              "sector": "Infrastructure"},
    "POWERGRID.NS":     {"company": "Power Grid",                   "sector": "Infrastructure"},
    "ADANIGREEN.NS":    {"company": "Adani Green Energy",           "sector": "Infrastructure"},
    "SIEMENS.NS":       {"company": "Siemens India",                "sector": "Infrastructure"},
    "ABB.NS":           {"company": "ABB India",                    "sector": "Infrastructure"},
    "HAVELLS.NS":       {"company": "Havells India",                "sector": "Infrastructure"},
    "CUMMINSIND.NS":    {"company": "Cummins India",                "sector": "Infrastructure"},
    "SUPREMEIND.NS":    {"company": "Supreme Industries",           "sector": "Infrastructure"},

    # ── Telecom ──────────────────────────────────────────────────────
    "BHARTIARTL.NS":    {"company": "Bharti Airtel",                "sector": "Telecom"},
    "INDUSTOWER.NS":    {"company": "Indus Towers",                 "sector": "Telecom"},
    "IDEA.NS":          {"company": "Vodafone Idea",                "sector": "Telecom"},

    # ── Real Estate ──────────────────────────────────────────────────
    "DLF.NS":           {"company": "DLF",                          "sector": "Real Estate"},
    "GODREJPROP.NS":    {"company": "Godrej Properties",            "sector": "Real Estate"},
    "OBEROIRLTY.NS":    {"company": "Oberoi Realty",                "sector": "Real Estate"},
    "PRESTIGE.NS":      {"company": "Prestige Estates",             "sector": "Real Estate"},
    "PHOENIXLTD.NS":    {"company": "Phoenix Mills",                "sector": "Real Estate"},

    # ── Consumer Discretionary ───────────────────────────────────────
    "TITAN.NS":         {"company": "Titan Company",                "sector": "Consumer Discretionary"},
    "ASIANPAINT.NS":    {"company": "Asian Paints",                 "sector": "Consumer Discretionary"},
    "PIDILITIND.NS":    {"company": "Pidilite Industries",          "sector": "Consumer Discretionary"},
    "BERGEPAINT.NS":    {"company": "Berger Paints",                "sector": "Consumer Discretionary"},
    "TRENT.NS":         {"company": "Trent",                        "sector": "Consumer Discretionary"},
    "DMART.NS":         {"company": "Avenue Supermarts (DMart)",    "sector": "Consumer Discretionary"},
    "NYKAA.NS":         {"company": "FSN E-Commerce (Nykaa)",       "sector": "Consumer Discretionary"},
}

def get_company(ticker):
    return TICKERS.get(ticker, {}).get("company", ticker)

def get_sector(ticker):
    return TICKERS.get(ticker, {}).get("sector", "Unknown")

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
