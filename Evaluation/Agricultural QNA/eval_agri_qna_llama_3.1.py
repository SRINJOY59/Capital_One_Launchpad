import random
import os
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from groq import Groq
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
import re
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# Download required NLTK data
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)
# Configure Groq API key


# Get Groq API key from env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)
print("✅ Groq client initialized successfully")

# Load KisanVaani Agriculture Q&A dataset
print("Loading KisanVaani Agriculture Q&A English dataset...")
dataset = load_dataset("KisanVaani/agriculture-qa-english-only")
train_data = dataset["train"]
print(f"Dataset loaded with {len(train_data)} samples")

def get_llama_response(question, max_retries=3):
    """Get response from Llama via Groq API for agriculture questions"""
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in agriculture, farming practices, and crop management. Provide clear, accurate, and practical answers to agriculture-related questions. Focus on scientific information and best practices."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.2,  # Lower temperature for more focused answers
                max_tokens=400,   # Moderate length for detailed but concise answers
                top_p=0.9,
                stream=False,
                stop=None,
            )

            return chat_completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return "Error: Could not get response"
    return "Error: Could not get response"

def clean_text(text):
    """Clean and normalize text"""
    if text is None:
        return ""

    text = str(text).strip()
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    return text

def calculate_hallucination_score(candidate, reference, context=""):
    """
    Calculate hallucination score based on unsupported content
    Formula: 1 - (Unsupported Tokens / Total Tokens)
    Higher score = less hallucination
    """
    if not candidate or not reference:
        return 0.0

    # Tokenize and normalize
    candidate_tokens = set(word_tokenize(candidate.lower()))
    reference_tokens = set(word_tokenize(reference.lower()))
    context_tokens = set(word_tokenize(context.lower())) if context else set()

    # Remove common stop words and punctuation for better evaluation
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
    candidate_tokens = candidate_tokens - stop_words
    reference_tokens = reference_tokens - stop_words
    context_tokens = context_tokens - stop_words

    # Combine reference and context as supported content
    supported_tokens = reference_tokens | context_tokens

    # Calculate unsupported tokens
    total_candidate_tokens = len(candidate_tokens)
    if total_candidate_tokens == 0:
        return 0.0

    unsupported_tokens = len(candidate_tokens - supported_tokens)
    hallucination_score = 1 - (unsupported_tokens / total_candidate_tokens)

    return max(0.0, hallucination_score)

def calculate_rouge_scores(candidate, reference):
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores"""
    if not candidate or not reference:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def calculate_bleu_score(candidate, reference):
    """
    Calculate BLEU score (1-gram to 4-gram)
    Formula: BLEU = BP * exp(sum(wn * log(pn)))
    """
    if not candidate or not reference:
        return 0.0

    candidate_tokens = word_tokenize(candidate.lower())
    reference_tokens = [word_tokenize(reference.lower())]

    # Use smoothing to handle zero matches
    smoothing_function = SmoothingFunction().method1

    try:
        bleu_score = sentence_bleu(
            reference_tokens,
            candidate_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing_function
        )
        return bleu_score
    except:
        return 0.0

def calculate_precision_recall_f1(candidate, reference):
    """Calculate token-level precision, recall, and F1 score"""
    if not candidate or not reference:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    candidate_tokens = Counter(word_tokenize(candidate.lower()))
    reference_tokens = Counter(word_tokenize(reference.lower()))

    # Calculate intersection
    overlap = candidate_tokens & reference_tokens
    overlap_count = sum(overlap.values())

    candidate_count = sum(candidate_tokens.values())
    reference_count = sum(reference_tokens.values())

    # Calculate metrics
    precision = overlap_count / candidate_count if candidate_count > 0 else 0.0
    recall = overlap_count / reference_count if reference_count > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_round_comprehensive(data, num_samples=10):
    """Evaluate one round with comprehensive metrics for KisanVaani dataset"""
    # Filter valid samples
    valid_samples = []
    for sample in data:
        question = sample.get("question")
        answer = sample.get("answers")  # Note: 'answers' not 'answer'

        if (question is not None and answer is not None and
            len(str(question).strip()) > 10 and
            len(str(answer).strip()) > 5):  # Lower threshold for shorter answers
            valid_samples.append(sample)

    if len(valid_samples) < num_samples:
        num_samples = len(valid_samples)
        print(f"Warning: Only {num_samples} valid samples available")

    samples = random.sample(valid_samples, num_samples)

    # Data storage
    ragas_data = {
        'question': [],
        'answer': [],           # Model generated
        'contexts': [],         # Question as context
        'ground_truth': []      # Original answer
    }

    overlap_metrics = {
        'hallucination_scores': [],
        'rouge_scores': [],
        'bleu_scores': [],
        'precision_recall_f1': []
    }

    print(f"Evaluating {num_samples} agriculture Q&A samples with comprehensive metrics...")

    for i, example in enumerate(samples, 1):
        question = clean_text(example.get("question"))
        ground_truth_answer = clean_text(example.get("answers"))  # Note: 'answers' key

        if not question or not ground_truth_answer:
            print(f"Skipping sample {i} due to empty question or answer")
            continue

        print(f"\nProcessing question {i}/{num_samples}:")
        print(f"Q: {question}")
        print(f"Expected: {ground_truth_answer}")

        # Get model response
        model_answer = get_llama_response(question)
        print(f"Model: {model_answer}")

        # Store RAGAS data
        ragas_data['question'].append(question)
        ragas_data['answer'].append(model_answer)
        ragas_data['contexts'].append([question])  # Question as context
        ragas_data['ground_truth'].append(ground_truth_answer)

        # Calculate overlap metrics
        hallucination_score = calculate_hallucination_score(
            model_answer, ground_truth_answer, question
        )
        rouge_scores = calculate_rouge_scores(model_answer, ground_truth_answer)
        bleu_score = calculate_bleu_score(model_answer, ground_truth_answer)
        prf_scores = calculate_precision_recall_f1(model_answer, ground_truth_answer)

        overlap_metrics['hallucination_scores'].append(hallucination_score)
        overlap_metrics['rouge_scores'].append(rouge_scores)
        overlap_metrics['bleu_scores'].append(bleu_score)
        overlap_metrics['precision_recall_f1'].append(prf_scores)

        # Display metrics for this sample
        print(f"🔍 Sample Metrics:")
        print(f"  Hallucination: {hallucination_score:.3f}")
        print(f"  ROUGE-1: {rouge_scores['rouge1']:.3f}")
        print(f"  ROUGE-L: {rouge_scores['rougeL']:.3f}")
        print(f"  BLEU: {bleu_score:.3f}")
        print(f"  F1: {prf_scores['f1']:.3f}")
        print("-" * 80)

    # Run RAGAS evaluation
    ragas_scores = None
    if ragas_data['question']:
        try:
            ragas_dataset = Dataset.from_dict(ragas_data)
            print("\n🔍 Running RAGAS evaluation...")
            ragas_result = evaluate(
                ragas_dataset,
                metrics=[
                    answer_relevancy,
                    faithfulness,
                    answer_correctness,
                    answer_similarity
                ],
            )
            ragas_scores = ragas_result
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")

    return ragas_scores, overlap_metrics

def summarize_overlap_metrics(overlap_metrics):
    """Calculate average scores for overlap metrics"""
    summary = {}

    if overlap_metrics['hallucination_scores']:
        summary['avg_hallucination_score'] = np.mean(overlap_metrics['hallucination_scores'])

    if overlap_metrics['rouge_scores']:
        rouge1_scores = [score['rouge1'] for score in overlap_metrics['rouge_scores']]
        rouge2_scores = [score['rouge2'] for score in overlap_metrics['rouge_scores']]
        rougeL_scores = [score['rougeL'] for score in overlap_metrics['rouge_scores']]

        summary['avg_rouge1'] = np.mean(rouge1_scores)
        summary['avg_rouge2'] = np.mean(rouge2_scores)
        summary['avg_rougeL'] = np.mean(rougeL_scores)

    if overlap_metrics['bleu_scores']:
        summary['avg_bleu'] = np.mean(overlap_metrics['bleu_scores'])

    if overlap_metrics['precision_recall_f1']:
        precision_scores = [score['precision'] for score in overlap_metrics['precision_recall_f1']]
        recall_scores = [score['recall'] for score in overlap_metrics['precision_recall_f1']]
        f1_scores = [score['f1'] for score in overlap_metrics['precision_recall_f1']]

        summary['avg_precision'] = np.mean(precision_scores)
        summary['avg_recall'] = np.mean(recall_scores)
        summary['avg_f1'] = np.mean(f1_scores)

    return summary

def main():
    """Main evaluation function for KisanVaani agriculture dataset"""
    print("🚀 KisanVaani Agriculture Q&A Comprehensive Evaluation")
    print("🌾 Dataset: agriculture-qa-english-only")
    print("📊 Metrics: RAGAS + Hallucination + ROUGE + BLEU + Precision/Recall/F1")
    print("=" * 80)

    # Test API connection
    try:
        test_response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3.1-8b-instant",
            max_tokens=10
        )
        print("✅ Groq API connection successful")
    except Exception as e:
        print(f"❌ Groq API connection failed: {e}")
        return

    # Show dataset sample
    print(f"\n📋 Dataset Info:")
    print(f"Total samples: {len(train_data)}")
    sample = train_data[0]
    print(f"Sample question: {sample['question']}")
    print(f"Sample answer: {sample['answers']}")

    # Run evaluation rounds
    all_ragas_scores = []
    all_overlap_metrics = []

    for round_num in range(3):
        print(f"\n🔄 === Round {round_num + 1} ===")
        ragas_scores, overlap_metrics = evaluate_round_comprehensive(train_data, num_samples=5)  # Smaller sample for demo

        if ragas_scores is not None:
            all_ragas_scores.append(ragas_scores)

        all_overlap_metrics.append(overlap_metrics)

        # Summarize round results
        overlap_summary = summarize_overlap_metrics(overlap_metrics)

        print(f"\n📊 Round {round_num + 1} Summary:")

        if ragas_scores is not None:
            print("🎯 RAGAS Metrics:")
            for metric, score in ragas_scores.items():
                if isinstance(score, (int, float)):
                    print(f"  {metric}: {score:.3f}")

        print("📈 Overlap Metrics:")
        for metric, value in overlap_summary.items():
            print(f"  {metric}: {value:.3f}")

    # Final comprehensive results
    print("\n" + "=" * 80)
    print("📈 FINAL COMPREHENSIVE RESULTS - KisanVaani Agriculture Q&A")
    print("=" * 80)

    # RAGAS averages
    if all_ragas_scores:
        print("🎯 AVERAGE RAGAS METRICS:")
        avg_ragas = {}
        for metric in all_ragas_scores[0].keys():
            if isinstance(all_ragas_scores[0][metric], (int, float)):
                scores = [rs[metric] for rs in all_ragas_scores if isinstance(rs[metric], (int, float))]
                if scores:
                    avg_ragas[metric] = np.mean(scores)

        for metric, score in avg_ragas.items():
            print(f"  {metric}: {score:.3f}")

    # Overlap metrics averages
    print("\n📈 AVERAGE OVERLAP METRICS:")
    combined_overlap = {
        'hallucination_scores': [],
        'rouge_scores': [],
        'bleu_scores': [],
        'precision_recall_f1': []
    }

    for metrics in all_overlap_metrics:
        combined_overlap['hallucination_scores'].extend(metrics['hallucination_scores'])
        combined_overlap['rouge_scores'].extend(metrics['rouge_scores'])
        combined_overlap['bleu_scores'].extend(metrics['bleu_scores'])
        combined_overlap['precision_recall_f1'].extend(metrics['precision_recall_f1'])

    final_overlap_summary = summarize_overlap_metrics(combined_overlap)

    for metric, value in final_overlap_summary.items():
        print(f"  {metric}: {value:.3f}")

    # Performance interpretation
    print("\n🏆 PERFORMANCE INTERPRETATION:")
    if final_overlap_summary:
        halluc_score = final_overlap_summary.get('avg_hallucination_score', 0)
        rouge1_score = final_overlap_summary.get('avg_rouge1', 0)
        bleu_score = final_overlap_summary.get('avg_bleu', 0)
        f1_score = final_overlap_summary.get('avg_f1', 0)

        print(f"🔸 Hallucination Control: {'Excellent' if halluc_score > 0.7 else 'Good' if halluc_score > 0.5 else 'Needs Improvement'} ({halluc_score:.3f})")
        print(f"🔸 Content Overlap (ROUGE-1): {'Excellent' if rouge1_score > 0.5 else 'Good' if rouge1_score > 0.3 else 'Needs Improvement'} ({rouge1_score:.3f})")
        print(f"🔸 Precision Match (BLEU): {'Excellent' if bleu_score > 0.4 else 'Good' if bleu_score > 0.2 else 'Needs Improvement'} ({bleu_score:.3f})")
        print(f"🔸 Overall F1: {'Excellent' if f1_score > 0.6 else 'Good' if f1_score > 0.4 else 'Needs Improvement'} ({f1_score:.3f})")

    print("\n📋 METRICS GUIDE:")
    print("🔹 Higher scores = Better performance for all metrics")
    print("🔹 Hallucination: 1.0 = No unsupported content")
    print("🔹 ROUGE: Recall-oriented overlap with reference")
    print("🔹 BLEU: Precision-oriented similarity")
    print("🔹 F1: Balance of precision and recall")

    print(f"\n🔧 CONFIGURATION:")
    print(f"🤖 Model: llama-3.1-8b-instant")
    print(f"🌡️  Temperature: 0.2 (focused responses)")
    print(f"🎯 Max Tokens: 400")
    print(f"📊 Dataset: KisanVaani/agriculture-qa-english-only")
    print(f"🔍 Total Samples Evaluated: {len(all_overlap_metrics) * 5}")

if __name__ == "__main__":
    main()