import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not loaded from .env")

print("API key loaded:", api_key[:7] + "..." + api_key[-4:])

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-5-mini",
    input="Say hello in one sentence."
)

print("\nOpenAI response:")
print(response.output_text)