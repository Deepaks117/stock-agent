from dotenv import load_dotenv
import os

load_dotenv()
print("NEWS_API_KEY:", os.getenv("NEWS_API_KEY"))