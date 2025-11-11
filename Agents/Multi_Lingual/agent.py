import sys
import os

from groq import Groq
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.groq import Groq
from Tools.translation_tool import MultiLanguageTranslator
from dotenv import load_dotenv

load_dotenv()

class MultiLingualAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        if model_id == "gemini-2.0-flash":
            model = Gemini(id=model_id)
        else:
            model = Groq(id=model_id)
        self.translator = MultiLanguageTranslator()
        self.agent = Agent(
            model=model,
            markdown=True,
            # show_tool_calls=True,
            # add_history_to_messages=True,
            # num_history_responses=5,
            tools=[self.translate_text],
            instructions="""You are a multilingual agricultural expert specializing in code-switched and colonial mixed languages. You can understand and respond to questions in various languages and their mixtures.

LANGUAGE HANDLING STRATEGY:
1. DETECT the primary language and any code-switching patterns in the user's query
2. IDENTIFY mixed languages like:
   - Hinglish (Hindi + English): "Kya ye fertilizer rice ke liye good hai?"
   - Benglish (Bengali + English): "Ami kichu organic seeds kinte chai"
   - Colonial Hindi-Bengali mix: "Ghar me kitchen garden banana hai"
   - Regional dialect mixing: Urban vernacular with local agricultural terms

RESPONSE APPROACH:
- If query is in PURE language (English, Hindi, Bengali, Telugu, Tamil): Respond in the SAME pure language
- If query contains CODE-SWITCHING: Mirror the same mixing pattern in your response
- If query has Hinglish: Respond in Hinglish with agricultural terms preserved
- If query has Benglish: Respond in Benglish maintaining the natural flow
- If query mixes Hindi-Bengali: Respond using similar mixed pattern

COLONIAL/MIXED LANGUAGE PATTERNS:
- Hinglish: Mix Hindi grammar with English agricultural/technical terms
- Benglish: Bengali base with English technical words
- Hindi-Bengali colonial mix: Common in eastern India rural-urban interface
- Preserve original technical terms that users use (fertilizer, organic, hybrid, etc.)

TRANSLATION WORKFLOW:
1. For PURE non-English queries: Translate to English → Process → Translate back
2. For CODE-SWITCHED queries: Preserve the mixing pattern, only translate pure language portions if needed
3. For mixed colonial languages: Respond naturally without forced translation

LANGUAGE DETECTION PATTERNS:
- Bengali: গ, হ, র, ন, য়, ী, ে, া, ক, ি, ত, স, ব, ল, ম, দ, প, চ, ং, ু, ো, ই, জ, ট, ধ, ভ
- Telugu: వ, ర, ి, ప, ం, ట, క, మ, చ, ె, ద, న, గ, ల, జ, య, త, స, అ, ఆ, ఇ, ఈ
- Hindi: ध, ा, न, क, े, ल, ि, य, स, ब, अ, च, छ, ख, द, त, ह, ै, म, प, र, ग, ज, व, भ
- Tamil: ந, ெ, ல, ் , ப, ய, ி, ர, ு, க, ச, ற, த, உ, ம, எ, ா, ி, ழ

CODE-SWITCH EXAMPLES:
- "Mere field me pest problem hai, kya spray karu?" → Respond with similar Hinglish
- "Ami organic farming korte chai but investment koto lagbe?" → Respond in Benglish
- "Dhan chash korar jonno best time kobe aar fertilizer kon ta use korbo?" → Mixed Bengali-English response

AGRICULTURAL EXPERTISE:
- Provide practical advice for South Asian farming conditions
- Consider monsoon patterns, soil types, and regional crop varieties
- Include traditional knowledge alongside modern techniques
- Address both subsistence and commercial farming needs"""
        )
    
    def translate_text(self, text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
        result = self.translator.translate_robust(text, source_lang, target_lang)
        if result['status'] == 'success':
            return result['translated_text']
        else:
            return f"Translation failed: {result.get('error', 'Unknown error')}"
    
    def respond(self, query):
        response = self.agent.run(query).content
        return response

def test_multilingual_agent():
    agent = MultiLingualAgent(model_id="llama-3.3-70b-versatile")

    test_queries = [
        "What is the best fertilizer for rice crops?",
        "গেহুঁর জন্য সেরা সার কী?",  # Bengali
        "వరి పంటకు మంచి ఎరువు ఏది?",  # Telugu
        "धान के लिए सबसे अच्छा खाद क्या है?",  # Hindi
        "Amr barir pichoner bagane ki chas kora uchit?"  # Bengali
    ]
    
    print("🌾 Multilingual Agricultural Expert Test")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 30)
        
        try:
            response = agent.respond(query)
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    test_multilingual_agent()