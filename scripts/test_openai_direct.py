from openai import OpenAI

OPENAI_API_KEY = "PASTE_YOUR_NEW_API_KEY_HERE"

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model="gpt-5-mini",
    input="Say hello in one short sentence."
)

print("SUCCESS!")
print(response.output_text)