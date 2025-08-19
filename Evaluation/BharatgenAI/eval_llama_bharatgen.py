import random
import os
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import os
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Get Groq API key from env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)
print("✅ Groq client initialized successfully")
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

def get_llama_response(prompt, max_retries=3):
    """Get response from Llama via Groq API with retry logic"""
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in agriculture. Answer multiple choice questions by providing only the exact text of the correct option. Do not include explanations, reasoning, or additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",  # Groq's Llama 3.3 70B model
                temperature=0.1,
                max_tokens=50,
                top_p=1,
                stream=False,
                stop=None,
            )

            return chat_completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return "Error: Could not get response"
    return "Error: Could not get response"

def find_best_match(generated_text, options):
    """Find the best matching option from the generated text"""
    generated_lower = generated_text.lower()

    # First, try to find exact matches
    for opt in options:
        if opt.lower() in generated_lower:
            return opt

    # If no exact match, find the option with most word overlap
    best_match = options[0]
    max_overlap = 0

    for opt in options:
        opt_words = set(opt.lower().split())
        generated_words = set(generated_lower.split())
        overlap = len(opt_words.intersection(generated_words))

        if overlap > max_overlap:
            max_overlap = overlap
            best_match = opt

    return best_match

def evaluate_round(data, num_samples=10):
    """Evaluate one round with specified number of samples"""
    samples = random.sample(list(data), num_samples)
    y_true = []
    y_pred = []

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

        # Map letter to actual option text
        option_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        correct_answer = options[option_map[correct_answer_letter]]

        # Prepare prompt for Llama
        options_str = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(options)])  # A, B, C, D format

        prompt = f"""Answer the following multiple choice question by providing only the exact text of the correct option.

Question: {question}

Options:
{options_str}

Instructions:
- Read the question carefully about agriculture/farming
- Consider each option
- Respond with only the exact text of the correct option
- Do not include option letters, numbers, explanations, or additional text

Answer:"""

        # Get model prediction
        print(f"Processing question {i}/{num_samples}...")
        output = get_llama_response(prompt)

        # Find the best matching option
        predicted_answer = find_best_match(output, options)

        y_true.append(correct_answer.strip())
        y_pred.append(predicted_answer.strip())

        # Display results
        print(f"Q: {question}")
        print(f"Options: {options}")
        print(f"Generated: {output}")
        print(f"Predicted: {predicted_answer}")
        print(f"Correct: {correct_answer} (Option {correct_answer_letter})")
        print(f"✅ Correct" if predicted_answer.strip() == correct_answer.strip() else "❌ Incorrect")
        print("-" * 80)

    # Compute accuracy
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred

def main():
    """Main evaluation function"""
    print("🚀 Starting BhashaBench-Krishi Evaluation with Llama 3.3 70B Instruct via Groq")
    print("=" * 70)

    # Test API connection
    try:
        test_response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3.3-70b-versatile",
            max_tokens=10
        )
        print("✅ Groq API connection successful")
    except Exception as e:
        print(f"❌ Groq API connection failed: {e}")
        print("Please check your API key and internet connection")
        return

    # Run 3 rounds of evaluation
    round_accuracies = []
    all_predictions = []

    for round_num in range(3):
        print(f"\n🔄 === Round {round_num + 1} ===")
        acc, y_true, y_pred = evaluate_round(train_data, num_samples=10)
        round_accuracies.append(acc)
        all_predictions.append((y_true, y_pred))

        print(f"📊 Accuracy for Round {round_num + 1}: {acc:.2%} ({acc:.4f})")
        print(f"✅ Correct: {sum(1 for t, p in zip(y_true, y_pred) if t == p)}/10")
        print(f"❌ Incorrect: {sum(1 for t, p in zip(y_true, y_pred) if t != p)}/10")

    # Final results
    print("\n" + "=" * 70)
    print("=" * 70)
    print("Round-wise Accuracies:")
    for i, acc in enumerate(round_accuracies, 1):
        print(f"  Round {i}: {acc:.2%}")

    avg_accuracy = sum(round_accuracies) / len(round_accuracies)
    print(f"\n🎯 Average Accuracy: {avg_accuracy:.2%} ({avg_accuracy:.4f})")
    print(f"📊 Standard Deviation: {(sum((acc - avg_accuracy)**2 for acc in round_accuracies) / len(round_accuracies))**0.5:.4f}")

    # Detailed breakdown
    total_correct = sum(sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
                       for true_labels, pred_labels in all_predictions)
    total_questions = len(round_accuracies) * 10

    print(f"📋 Total Questions: {total_questions}")
    print(f"✅ Total Correct: {total_correct}")
    print(f"❌ Total Incorrect: {total_questions - total_correct}")

    print(f"🌡️  Temperature: 0.1")
    print(f"🎯 Max Tokens: 50")

if __name__ == "__main__":
    main()