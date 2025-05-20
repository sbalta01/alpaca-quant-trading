from alpaca.trading.client import TradingClient
from alpaca.trading.enums import Orderside, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream


# ── CONFIG ─────────────────────────────────────────────────────────
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path('.') / '.env')


API_KEY    = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

client = TradingClient(API_KEY, API_SECRET, paper = True)
account = dict(client.get_account())
for k,v in account.items():
    print(f"{k:30}{v}")
