import google.generativeai as genai
import os
import toml

# Try to load secrets
try:
    secrets = toml.load(".streamlit/secrets.toml")
    api_key = secrets.get("GOOGLE_API_KEY")
except Exception as e:
    print(f"Error loading secrets: {e}")
    api_key = None

if not api_key:
    api_key = "AIzaSyCStwhPDY3-E24AzTT1GOEIvK5X96g7Er8"

genai.configure(api_key=api_key)

model_name = 'gemini-flash-latest'
print(f"Testing model: {model_name}")

try:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content("Hello, are you working?")
    print(f"Response: {response.text}")
    print("SUCCESS")
except Exception as e:
    print(f"Error generating content: {e}")
