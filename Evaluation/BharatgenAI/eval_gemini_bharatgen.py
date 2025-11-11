import random
import os
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import google.generativeai as genai
from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Configure Google API key
GOOGLE_API_KEY = "your_google_api_key_here"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
print("✅ Google API key configured successfully")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)
    print("✅ Hugging Face login successful")
else:
    print("⚠️ Hugging Face token not found. Proceeding without login...")

# Load dataset (English config)
print("Loading BhashaBench-Krishi dataset...")
dataset = load_dataset("bharatgenai/BhashaBench-Krishi", "English")
train_data = dataset["test"]
print(f"Dataset loaded with {len(train_data)} samples")

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash-exp')
print("✅ Gemini model initialized")

def get_gemini_response(prompt, max_retries=3):
    """Get response from Gemini with retry logic"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent responses
                    max_output_tokens=100,  # Increased for better responses
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return "Error: Could not get response"
    return "Error: Could not get response"

def extract_option_from_response(response_text):
    """Extract option (A, B, C, D) from model response"""
    import re

    response_upper = response_text.upper()

    # Look for patterns like "A)", "A.", "(A)", "Option A", etc.
    patterns = [
        r'\b([ABCD])\)',  # A), B), etc.
        r'\b([ABCD])\.',  # A., B., etc.
        r'\(([ABCD])\)',  # (A), (B), etc.
        r'OPTION\s+([ABCD])',  # OPTION A, OPTION B, etc.
        r'ANSWER\s*:?\s*([ABCD])',  # ANSWER: A, ANSWER A, etc.
        r'\b([ABCD])\b(?=\s|$)'  # Standalone A, B, C, D
    ]

    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)

    return None

def evaluate_round(data, num_samples=10):
    """Evaluate one round with specified number of samples"""
    samples = random.sample(list(data), num_samples)
    correct_count = 0
    results = []

    print(f"Evaluating {num_samples} samples...")

    for i, example in enumerate(samples, 1):
        question = example["question"]
        # Extract options from individual columns
        options = [
            example["option_a"],
            example["option_b"],
            example["option_c"],
            example["option_d"]
        ]
        correct_answer_letter = example["correct_answer"]  # This will be 'A', 'B', 'C', or 'D'

        # Prepare prompt for Gemini
        options_str = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(options)])

        prompt = f"""You are an expert in agriculture. Answer the following multiple choice question.

Question: {question}

Options:
{options_str}

Instructions:
- Read the question carefully
- Consider each option
- Respond with the letter of the correct option (A, B, C, or D) followed by your reasoning
- Format: "Answer: [LETTER]" then explain why

Answer:"""

        # Get model prediction
        print(f"Processing question {i}/{num_samples}...")
        output = get_gemini_response(prompt)

        # Extract option letter from response
        predicted_letter = extract_option_from_response(output)

        # Check if prediction is correct
        is_correct = predicted_letter == correct_answer_letter
        if is_correct:
            correct_count += 1

        # Store result
        result = {
            'question_num': i,
            'question': question,
            'options': options,
            'model_response': output,
            'predicted_letter': predicted_letter,
            'correct_letter': correct_answer_letter,
            'is_correct': is_correct
        }
        results.append(result)

        # Display results
        print(f"Q{i}: {question}")
        print(f"Options:")
        for j, opt in enumerate(options):
            print(f"  {chr(65+j)}. {opt}")
        print(f"Model Response: {output}")
        print(f"Predicted: {predicted_letter}")
        print(f"Correct: {correct_answer_letter}")

        if is_correct:
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
        print("-" * 80)

    accuracy = correct_count / num_samples
    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': num_samples,
        'detailed_results': results
    }

def main():
    """Main evaluation function"""
    print("🚀 Starting BhashaBench-Krishi Evaluation with Gemini 2.0 Flash")
    print("=" * 80)

    # Run 3 rounds of evaluation
    all_results = []
    all_detailed_results = []

    for round_num in range(3):
        print(f"\n🔄 === Round {round_num + 1} ===")
        results = evaluate_round(train_data, num_samples=10)
        all_results.append(results)
        all_detailed_results.extend(results['detailed_results'])

        print(f"\n📊 Results for Round {round_num + 1}:")
        print(f"  Accuracy: {results['accuracy']:.2%} ({results['correct_count']}/{results['total_count']})")

    # Final results
    print("\n" + "=" * 80)
    print("📈 FINAL RESULTS")
    print("=" * 80)

    # Calculate overall accuracy
    total_correct = sum(r['correct_count'] for r in all_results)
    total_questions = sum(r['total_count'] for r in all_results)
    overall_accuracy = total_correct / total_questions

    print("📊 OVERALL ACCURACY:")
    print(f"  Total Correct: {total_correct}")
    print(f"  Total Questions: {total_questions}")
    print(f"  Accuracy: {overall_accuracy:.2%} ({overall_accuracy:.4f})")

    print("\n📋 ROUND-WISE BREAKDOWN:")
    for i, results in enumerate(all_results, 1):
        print(f"  Round {i}: {results['accuracy']:.2%} ({results['correct_count']}/{results['total_count']})")

    # Show some examples of correct and incorrect predictions
    correct_examples = [r for r in all_detailed_results if r['is_correct']]
    incorrect_examples = [r for r in all_detailed_results if not r['is_correct']]

    if correct_examples:
        print(f"\n✅ EXAMPLE CORRECT PREDICTIONS:")
        for i, example in enumerate(correct_examples[:3], 1):  # Show first 3
            print(f"  {i}. Q: {example['question'][:50]}...")
            print(f"     Predicted: {example['predicted_letter']} | Correct: {example['correct_letter']}")

    if incorrect_examples:
        print(f"\n❌ EXAMPLE INCORRECT PREDICTIONS:")
        for i, example in enumerate(incorrect_examples[:3], 1):  # Show first 3
            print(f"  {i}. Q: {example['question'][:50]}...")
            print(f"     Predicted: {example['predicted_letter']} | Correct: {example['correct_letter']}")

    print(f"\n🎯 FINAL ACCURACY: {overall_accuracy:.2%}")

if __name__ == "__main__":
    main()