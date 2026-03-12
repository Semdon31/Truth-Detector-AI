from dotenv import load_dotenv
from openai import OpenAI
import os 

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def analyze_deception(text):

    prompt = f"""
    Analyze the folllowing statement for potential deception.

    Statement: "{text}"

    Look for:
    - distancing language
    - vagueness
    - emotional manipulation
    - unusual phrasing

    Return a short explanation.

    """
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content