import json
import logging
from typing import Dict, List, Optional
from deep_translator import GoogleTranslator
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiLanguageTranslator:
    def __init__(self):
        self.supported_languages = {
            'auto': 'Auto-detect',
            'en': 'English',
            'hi': 'Hindi (हिंदी)',
            'te': 'Telugu (తెలుగు)',
            'ta': 'Tamil (தமிழ்)',
            'kn': 'Kannada (ಕನ್ನಡ)',
            'ml': 'Malayalam (മലയാളം)',
            'bn': 'Bengali (বাংলা)',
            'gu': 'Gujarati (ગુજરાતી)',
            'mr': 'Marathi (मराठी)',
            'pa': 'Punjabi (ਪੰਜਾਬੀ)',
            'or': 'Odia (ଓଡ଼ିଆ)',
            'as': 'Assamese (অসমীয়া)',
            'ur': 'Urdu (اردو)',
            'ne': 'Nepali (नेपाली)',
            'si': 'Sinhala (සිංහල)',
            'my': 'Myanmar (မြန်မာ)',
            'th': 'Thai (ไทย)',
            'vi': 'Vietnamese (Tiếng Việt)',
            'ko': 'Korean (한국어)',
            'ja': 'Japanese (日本語)',
            'zh': 'Chinese Simplified (中文)',
            'zh-tw': 'Chinese Traditional (繁體中文)',
            'fr': 'French (Français)',
            'de': 'German (Deutsch)',
            'es': 'Spanish (Español)',
            'pt': 'Portuguese (Português)',
            'ru': 'Russian (Русский)',
            'ar': 'Arabic (العربية)',
            'tr': 'Turkish (Türkçe)',
            'it': 'Italian (Italiano)',
            'nl': 'Dutch (Nederlands)',
            'sv': 'Swedish (Svenska)',
            'da': 'Danish (Dansk)',
            'no': 'Norwegian (Norsk)',
            'fi': 'Finnish (Suomi)',
            'pl': 'Polish (Polski)',
            'cs': 'Czech (Čeština)',
            'sk': 'Slovak (Slovenčina)',
            'hu': 'Hungarian (Magyar)',
            'ro': 'Romanian (Română)',
            'bg': 'Bulgarian (Български)',
            'hr': 'Croatian (Hrvatski)',
            'sr': 'Serbian (Српски)',
            'sl': 'Slovenian (Slovenščina)',
            'et': 'Estonian (Eesti)',
            'lv': 'Latvian (Latviešu)',
            'lt': 'Lithuanian (Lietuvių)',
            'mt': 'Maltese (Malti)',
            'ga': 'Irish (Gaeilge)',
            'cy': 'Welsh (Cymraeg)',
            'eu': 'Basque (Euskera)',
            'ca': 'Catalan (Català)',
            'gl': 'Galician (Galego)',
            'is': 'Icelandic (Íslenska)',
            'sq': 'Albanian (Shqip)',
            'mk': 'Macedonian (Македонски)',
            'el': 'Greek (Ελληνικά)',
            'he': 'Hebrew (עברית)',
            'fa': 'Persian (فارسی)',
            'sw': 'Swahili (Kiswahili)',
            'zu': 'Zulu (isiZulu)',
            'af': 'Afrikaans',
            'am': 'Amharic (አማርኛ)',
            'az': 'Azerbaijani (Azərbaycan)',
            'be': 'Belarusian (Беларуская)',
            'bs': 'Bosnian (Bosanski)',
            'eu': 'Basque (Euskera)',
            'ka': 'Georgian (ქართული)',
            'hy': 'Armenian (Հայերեն)',
            'is': 'Icelandic (Íslenska)',
            'kk': 'Kazakh (Қазақ)',
            'ky': 'Kyrgyz (Кыргыз)',
            'lo': 'Lao (ລາວ)',
            'lv': 'Latvian (Latviešu)',
            'lt': 'Lithuanian (Lietuvių)',
            'lb': 'Luxembourgish (Lëtzebuergesch)',
            'ms': 'Malay (Bahasa Melayu)',
            'mn': 'Mongolian (Монгол)',
            'ps': 'Pashto (پښتو)',
            'tg': 'Tajik (Тоҷикӣ)',
            'tk': 'Turkmen (Türkmen)',
            'uk': 'Ukrainian (Українська)',
            'uz': 'Uzbek (Oʻzbek)',
            'yi': 'Yiddish (ייִדיש)'
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        return self.supported_languages
    
    def translate_robust(self, text: str, source_lang: str = 'auto', target_lang: str = 'te') -> Dict:
        if target_lang not in self.supported_languages:
            return {
                "error": f"Unsupported target language: {target_lang}",
                "supported_languages": list(self.supported_languages.keys())
            }
        
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(text)
            
            return {
                "original_text": text,
                "translated_text": translated,
                "source_language": self.supported_languages.get(source_lang, source_lang),
                "target_language": self.supported_languages.get(target_lang, target_lang),
                "status": "success",
                "method": "deep_translator"
            }
            
        except Exception as e:
            logger.warning(f"deep-translator failed: {e}, trying httpx fallback")
            
            try:
                url = "https://translate.googleapis.com/translate_a/single"
                params = {
                    "client": "gtx",
                    "sl": source_lang,
                    "tl": target_lang,
                    "dt": "t",
                    "q": text
                }
                
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    translated_text = ''.join([sentence[0] for sentence in data[0]])
                    
                    return {
                        "original_text": text,
                        "translated_text": translated_text,
                        "source_language": self.supported_languages.get(source_lang, source_lang),
                        "target_language": self.supported_languages.get(target_lang, target_lang),
                        "status": "success",
                        "method": "httpx_fallback"
                    }
                    
            except Exception as fallback_error:
                logger.error(f"All translation methods failed: {fallback_error}")
                
                return {
                    "original_text": text,
                    "translated_text": f"Translation failed: {text}",
                    "source_language": self.supported_languages.get(source_lang, source_lang),
                    "target_language": self.supported_languages.get(target_lang, target_lang),
                    "status": "error",
                    "error": str(fallback_error),
                    "method": "error_fallback"
                }
    
    def translate_file(self, file_path: str, source_lang: str = 'auto', target_lang: str = 'te') -> Dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            result = self.translate_robust(content, source_lang, target_lang)
            
            if result['status'] == 'success':
                output_file = f"{file_path}_{target_lang}_translated.txt"
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(result['translated_text'])
                result['output_file'] = output_file
            
            return result
            
        except Exception as e:
            return {
                "error": f"Failed to read file {file_path}: {str(e)}",
                "status": "error"
            }
    
    def batch_translate(self, texts: List[str], source_lang: str = 'auto', target_lang: str = 'te') -> List[Dict]:
        results = []
        for i, text in enumerate(texts):
            print(f"Translating {i+1}/{len(texts)}...")
            result = self.translate_robust(text, source_lang, target_lang)
            results.append(result)
        return results
    
    def detect_language(self, text: str) -> Dict:
        try:
            translator = GoogleTranslator(source='auto', target='en')
            detected = translator.translate(text)
            
            return {
                "text": text,
                "detected_language": "auto-detected",
                "status": "success"
            }
        except Exception as e:
            return {
                "text": text,
                "error": str(e),
                "status": "error"
            }

def interactive_translation():
    translator = MultiLanguageTranslator()
    
    print("🌐 Multi-Language Translation Tool")
    print("=" * 50)
    
    while True:
        print("\n📋 Choose an option:")
        print("1. Single text translation")
        print("2. File translation")
        print("3. Batch translation")
        print("4. Show supported languages")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            single_text_translation(translator)
        elif choice == "2":
            file_translation(translator)
        elif choice == "3":
            batch_translation(translator)
        elif choice == "4":
            show_languages(translator)
        elif choice == "5":
            print("Goodbye! 👋")
            break
        else:
            print("❌ Invalid choice! Please enter 1-5.")

def single_text_translation(translator):
    print("\n📝 Single Text Translation")
    print("-" * 30)
    
    text = input("Enter text to translate: ").strip()
    if not text:
        print("❌ No text entered!")
        return
    
    print("\n🔤 Select source language (or 'auto' for auto-detect):")
    source_lang = input("Source language code (default: auto): ").strip() or 'auto'
    
    print("\n🎯 Select target language:")
    show_popular_languages()
    target_lang = input("Target language code (default: te): ").strip() or 'te'
    
    print("\n🔄 Translating...")
    result = translator.translate_robust(text, source_lang, target_lang)
    
    print("\n" + "=" * 60)
    print("📊 TRANSLATION RESULT")
    print("=" * 60)
    
    if result['status'] == 'success':
        print(f"✅ Status: {result['status']}")
        print(f"🔤 Source: {result['source_language']}")
        print(f"🎯 Target: {result['target_language']}")
        print(f"⚙️ Method: {result['method']}")
        print(f"\n📝 Original: {result['original_text']}")
        print(f"🌐 Translated: {result['translated_text']}")
    else:
        print(f"❌ Status: {result['status']}")
        print(f"🚫 Error: {result.get('error', 'Unknown error')}")

def file_translation(translator):
    print("\n📁 File Translation")
    print("-" * 20)
    
    file_path = input("Enter file path: ").strip().replace('"', '')
    if not file_path:
        print("❌ No file path entered!")
        return
    
    source_lang = input("Source language code (default: auto): ").strip() or 'auto'
    target_lang = input("Target language code (default: te): ").strip() or 'te'
    
    print("\n🔄 Translating file...")
    result = translator.translate_file(file_path, source_lang, target_lang)
    
    if result['status'] == 'success':
        print(f"✅ File translated successfully!")
        print(f"💾 Output saved to: {result['output_file']}")
    else:
        print(f"❌ Translation failed: {result.get('error', 'Unknown error')}")

def batch_translation(translator):
    print("\n📚 Batch Translation")
    print("-" * 20)
    
    print("Enter texts to translate (press Enter twice to finish):")
    texts = []
    while True:
        text = input(f"Text {len(texts)+1}: ").strip()
        if not text:
            break
        texts.append(text)
    
    if not texts:
        print("❌ No texts entered!")
        return
    
    source_lang = input("Source language code (default: auto): ").strip() or 'auto'
    target_lang = input("Target language code (default: te): ").strip() or 'te'
    
    print(f"\n🔄 Translating {len(texts)} texts...")
    results = translator.batch_translate(texts, source_lang, target_lang)
    
    print("\n" + "=" * 60)
    print("📊 BATCH TRANSLATION RESULTS")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Original: {result['original_text']}")
        if result['status'] == 'success':
            print(f"   Translated: {result['translated_text']}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")

def show_languages(translator):
    languages = translator.get_supported_languages()
    
    print("\n🌍 Supported Languages")
    print("=" * 50)
    
    for code, name in sorted(languages.items()):
        print(f"{code:6} - {name}")
    
    print(f"\nTotal: {len(languages)} languages supported")

def show_popular_languages():
    popular = {
        'auto': 'Auto-detect',
        'en': 'English',
        'bn': 'Bengali (বাংলা)',
        'hi': 'Hindi (हिंदी)',
        'te': 'Telugu (తెలుగు)',
        'ta': 'Tamil (தமிழ்)',
        'kn': 'Kannada (ಕನ್ನಡ)',
        'ml': 'Malayalam (മലയാളം)',
        'bn': 'Bengali (বাংলা)',
        'mr': 'Marathi (मराठी)',
        'gu': 'Gujarati (ગુજરાતી)',
        'ur': 'Urdu (اردو)',
        'fr': 'French (Français)',
        'de': 'German (Deutsch)',
        'es': 'Spanish (Español)',
        'zh': 'Chinese (中文)',
        'ja': 'Japanese (日本語)',
        'ko': 'Korean (한국어)',
        'ar': 'Arabic (العربية)',
        'ru': 'Russian (Русский)'
    }
    
    print("Popular languages:")
    for code, name in popular.items():
        print(f"  {code} - {name}")

def quick_translation():
    translator = MultiLanguageTranslator()

    hardcoded_texts = [
            "Good morning, how are you today?",
            "Agriculture is the backbone of our economy",
            "Please recommend the best fertilizer for wheat crops",
            "What is the current market price of rice?",
            "Sustainable farming practices help protect the environment"
        ]
        
       
    source = 'en'
    target = 'bn'
        
    for i, text in enumerate(hardcoded_texts, 1):
        print(f"\n{i}. Translating: '{text}'")
        result = translator.translate_robust(text, source, target)
        if result['status'] == 'success':
            print(f"   Telugu: {result['translated_text']}")
        else:
            print(f"   Error: {result.get('error', 'Translation failed')}")

if __name__ == "__main__":
    quick_translation()
