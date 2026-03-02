import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Load your .env file
load_dotenv()

# 2. Get API Key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file!")
else:
    try:
        genai.configure(api_key=api_key)
        print("--- List of Available Models ---")
        
        # 3. List models and their capabilities
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Model Name: {m.name}")
                print(f"Capabilities: {m.supported_generation_methods}")
                print("-" * 30)
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")