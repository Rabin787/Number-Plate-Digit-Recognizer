
import io
import os
from PIL import Image
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "models/gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

def classify_image_gemini(image_bytes: bytes) -> dict:
    try:
        # Convert bytes → PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        prompt = """
        You are a strict image classifier for Nepali vehicle number plate characters.

        Allowed classes ONLY:
        - Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        - Devanagari letters: बा, प

        Rules:
        1. Look carefully at the image.
        2. If the image clearly contains ONE of the allowed classes, return ONLY that exact class.
        3. If the image contains anything else (multiple characters, symbols, letters other than बा or प, words, unclear image, noise, background, or non-number-plate content), return EXACTLY:
        Unknown / Out of dataset
        4. Do NOT explain your answer.
        5. Do NOT add any extra text.

        Return format:
        - Just the class name OR "Unknown / Out of dataset"

        """

        #Gemini call
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image]   
        )

        return {
            "method": "agent",
            "prediction": response.text.strip()
        }

    except Exception as e:
        msg = str(e).lower()

        if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
            return {
                "method": "agent",
                "prediction": "⚠️ Gemini API quota exceeded. Try later or use ML model."
            }

        return {
            "method": "agent",
            "prediction": f"Error: {str(e)}"
        }
