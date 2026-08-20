import os
from dotenv import load_dotenv
from google import genai

# Load API keys from .env file
load_dotenv()

def verify_gemini_keys():
    print("=" * 45)
    print("   Gemini API Keys Verification   ")
    print("=" * 45 + "\n")

    for i in range(1, 6):
        key_name = f"GEMINI_API_KEY_{i}"
        api_key = os.getenv(key_name)

        if not api_key:
            print(f"❌ {key_name}: NOT FOUND in .env file")
            continue

        try:
            # Initialize Client with google-genai SDK
            client = genai.Client(api_key=api_key)
            
            # Send test request with active model
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents='Ping'
            )

            if response and response.text:
                masked_key = f"{api_key[:6]}...{api_key[-4:]}"
                print(f"✅ {key_name} ({masked_key}): Working!")
            else:
                print(f"⚠️ {key_name}: Connected, but returned empty text.")

        except Exception as e:
            print(f"❌ {key_name}: Failed -> {e}")

    print("\n" + "=" * 45)

if __name__ == "__main__":
    verify_gemini_keys()