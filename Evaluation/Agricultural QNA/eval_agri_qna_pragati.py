# import pandas as pd
# import numpy as np
# import random
# import time
# from collections import Counter
# from rouge_score import rouge_scorer
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.tokenize import word_tokenize
# import nltk

# # Import your workflow
# from workflow import run_workflow

# # Download required NLTK data
# for resource in ["punkt", "punkt_tab"]:
#     try:
#         nltk.data.find(f"tokenizers/{resource}")
#     except LookupError:
#         nltk.download(resource)

# class WorkflowEvaluator:
#     def __init__(self, dataset_path):
#         """Initialize evaluator with dataset"""
#         self.dataset_path = dataset_path
#         self.df = pd.read_csv(dataset_path)
#         print(f"Loaded dataset with {len(self.df)} samples")
        
#     def clean_text(self, text):
#         """Clean and normalize text"""
#         if text is None or pd.isna(text):
#             return ""
        
#         text = str(text).strip()
#         # Remove extra whitespace and normalize
#         text = ' '.join(text.split())
#         return text
    
#     def calculate_hallucination_score(self, candidate, reference, context=""):
#         """
#         Calculate hallucination score based on unsupported content
#         Formula: 1 - (Unsupported Tokens / Total Tokens)
#         Higher score = less hallucination
#         """
#         if not candidate or not reference:
#             return 0.0

#         # Tokenize and normalize
#         candidate_tokens = set(word_tokenize(candidate.lower()))
#         reference_tokens = set(word_tokenize(reference.lower()))
#         context_tokens = set(word_tokenize(context.lower())) if context else set()

#         # Remove common stop words and punctuation for better evaluation
#         stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
#         candidate_tokens = candidate_tokens - stop_words
#         reference_tokens = reference_tokens - stop_words
#         context_tokens = context_tokens - stop_words

#         # Combine reference and context as supported content
#         supported_tokens = reference_tokens | context_tokens

#         # Calculate unsupported tokens
#         total_candidate_tokens = len(candidate_tokens)
#         if total_candidate_tokens == 0:
#             return 0.0

#         unsupported_tokens = len(candidate_tokens - supported_tokens)
#         hallucination_score = 1 - (unsupported_tokens / total_candidate_tokens)

#         return max(0.0, hallucination_score)

#     def calculate_rouge_scores(self, candidate, reference):
#         """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores"""
#         if not candidate or not reference:
#             return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

#         scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#         scores = scorer.score(reference, candidate)

#         return {
#             'rouge1': scores['rouge1'].fmeasure,
#             'rouge2': scores['rouge2'].fmeasure,
#             'rougeL': scores['rougeL'].fmeasure
#         }

#     def calculate_bleu_score(self, candidate, reference):
#         """
#         Calculate BLEU score (1-gram to 4-gram)
#         Formula: BLEU = BP * exp(sum(wn * log(pn)))
#         """
#         if not candidate or not reference:
#             return 0.0

#         candidate_tokens = word_tokenize(candidate.lower())
#         reference_tokens = [word_tokenize(reference.lower())]

#         # Use smoothing to handle zero matches
#         smoothing_function = SmoothingFunction().method1

#         try:
#             bleu_score = sentence_bleu(
#                 reference_tokens,
#                 candidate_tokens,
#                 weights=(0.25, 0.25, 0.25, 0.25),
#                 smoothing_function=smoothing_function
#             )
#             return bleu_score
#         except:
#             return 0.0

#     def calculate_precision_recall_f1(self, candidate, reference):
#         """Calculate token-level precision, recall, and F1 score"""
#         if not candidate or not reference:
#             return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

#         candidate_tokens = Counter(word_tokenize(candidate.lower()))
#         reference_tokens = Counter(word_tokenize(reference.lower()))

#         # Calculate intersection
#         overlap = candidate_tokens & reference_tokens
#         overlap_count = sum(overlap.values())

#         candidate_count = sum(candidate_tokens.values())
#         reference_count = sum(reference_tokens.values())

#         # Calculate metrics
#         precision = overlap_count / candidate_count if candidate_count > 0 else 0.0
#         recall = overlap_count / reference_count if reference_count > 0 else 0.0
#         f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

#         return {
#             'precision': precision,
#             'recall': recall,
#             'f1': f1
#         }
    
#     def calculate_response_length_score(self, candidate, reference):
#         """Calculate length similarity score"""
#         if not candidate or not reference:
#             return 0.0
        
#         candidate_len = len(candidate.split())
#         reference_len = len(reference.split())
        
#         if reference_len == 0:
#             return 0.0
        
#         # Length ratio score (closer to 1.0 is better)
#         length_ratio = min(candidate_len, reference_len) / max(candidate_len, reference_len)
#         return length_ratio
    
#     def evaluate_sample(self, prompt, reference_response, mode="tooling"):
#         """Evaluate a single sample using the workflow"""
#         print(f"Processing prompt: {prompt[:100]}...")
        
#         # Get workflow response
#         start_time = time.time()
#         try:
#             result = run_workflow(prompt, mode, image_path=None)
#             workflow_response = result.get('answer', '')
#             processing_time = time.time() - start_time
            
#             # Extract additional workflow metrics
#             is_answer_complete = result.get('is_answer_complete', False)
#             final_mode = result.get('final_mode', mode)
#             switched_modes = result.get('switched_modes', False)
            
#         except Exception as e:
#             print(f"Error running workflow: {e}")
#             workflow_response = ""
#             processing_time = 0.0
#             is_answer_complete = False
#             final_mode = mode
#             switched_modes = False
        
#         # Clean texts
#         cleaned_workflow = self.clean_text(workflow_response)
#         cleaned_reference = self.clean_text(reference_response)
#         cleaned_prompt = self.clean_text(prompt)
        
#         # Calculate all metrics
#         metrics = {}
        
#         # Overlap metrics
#         metrics['hallucination_score'] = self.calculate_hallucination_score(
#             cleaned_workflow, cleaned_reference, cleaned_prompt
#         )
        
#         rouge_scores = self.calculate_rouge_scores(cleaned_workflow, cleaned_reference)
#         metrics.update(rouge_scores)
        
#         metrics['bleu_score'] = self.calculate_bleu_score(cleaned_workflow, cleaned_reference)
        
#         prf_scores = self.calculate_precision_recall_f1(cleaned_workflow, cleaned_reference)
#         metrics.update(prf_scores)
        
#         # Length similarity
#         metrics['length_score'] = self.calculate_response_length_score(cleaned_workflow, cleaned_reference)
        
#         # Workflow-specific metrics
#         metrics['processing_time'] = processing_time
#         metrics['is_answer_complete'] = is_answer_complete
#         metrics['final_mode'] = final_mode
#         metrics['switched_modes'] = switched_modes
        
#         # Response lengths
#         metrics['workflow_response_length'] = len(cleaned_workflow.split())
#         metrics['reference_response_length'] = len(cleaned_reference.split())
        
#         return {
#             'metrics': metrics,
#             'workflow_response': cleaned_workflow,
#             'reference_response': cleaned_reference,
#             'prompt': cleaned_prompt
#         }
    
#     def run_evaluation(self, num_samples=30, mode="tooling"):
#         """Run comprehensive evaluation on random samples"""
#         print(f"\n🚀 Agriculture Workflow Comprehensive Evaluation")
#         print(f"📊 Evaluating {num_samples} random samples")
#         print(f"🔧 Initial Mode: {mode}")
#         print("=" * 80)
        
#         # Sample random data
#         if num_samples > len(self.df):
#             num_samples = len(self.df)
#             print(f"Warning: Only {num_samples} samples available")
        
#         sample_indices = random.sample(range(len(self.df)), num_samples)
#         samples = self.df.iloc[sample_indices]
        
#         # Storage for results
#         all_metrics = []
#         detailed_results = []
        
#         # Process each sample
#         for idx, (_, row) in enumerate(samples.iterrows(), 1):
#             prompt = row['prompt']
#             reference = row['cleaned_response']
            
#             print(f"\n{'='*60}")
#             print(f"Sample {idx}/{num_samples}")
#             print(f"{'='*60}")
            
#             result = self.evaluate_sample(prompt, reference, mode)
            
#             all_metrics.append(result['metrics'])
#             detailed_results.append(result)
            
#             # Display sample metrics
#             metrics = result['metrics']
#             print(f"\n📊 Sample {idx} Metrics:")
#             print(f"  🔍 Hallucination Score: {metrics['hallucination_score']:.3f}")
#             print(f"  📝 ROUGE-1: {metrics['rouge1']:.3f}")
#             print(f"  📝 ROUGE-L: {metrics['rougeL']:.3f}")
#             print(f"  🎯 BLEU Score: {metrics['bleu_score']:.3f}")
#             print(f"  ⚖️  F1 Score: {metrics['f1']:.3f}")
#             print(f"  📏 Length Score: {metrics['length_score']:.3f}")
#             print(f"  ⏱️  Processing Time: {metrics['processing_time']:.2f}s")
#             print(f"  ✅ Answer Complete: {metrics['is_answer_complete']}")
#             print(f"  🔄 Final Mode: {metrics['final_mode']}")
#             print(f"  🔀 Switched Modes: {metrics['switched_modes']}")
            
#             print(f"\n📝 Response Preview:")
#             print(f"  Reference: {result['reference_response'][:150]}...")
#             print(f"  Workflow:  {result['workflow_response'][:150]}...")
        
#         return self.calculate_final_results(all_metrics, detailed_results)
    
#     def calculate_final_results(self, all_metrics, detailed_results):
#         """Calculate and display final comprehensive results"""
#         print("\n" + "=" * 80)
#         print("📈 FINAL COMPREHENSIVE RESULTS - Agriculture Workflow Evaluation")
#         print("=" * 80)
        
#         # Calculate averages for numerical metrics
#         numerical_metrics = [
#             'hallucination_score', 'rouge1', 'rouge2', 'rougeL', 'bleu_score',
#             'precision', 'recall', 'f1', 'length_score', 'processing_time',
#             'workflow_response_length', 'reference_response_length'
#         ]
        
#         avg_metrics = {}
#         for metric in numerical_metrics:
#             values = [m[metric] for m in all_metrics if metric in m]
#             if values:
#                 avg_metrics[f'avg_{metric}'] = np.mean(values)
#                 avg_metrics[f'std_{metric}'] = np.std(values)
        
#         # Calculate workflow-specific metrics
#         complete_answers = sum(1 for m in all_metrics if m.get('is_answer_complete', False))
#         mode_switches = sum(1 for m in all_metrics if m.get('switched_modes', False))
        
#         final_modes = [m.get('final_mode', 'unknown') for m in all_metrics]
#         mode_distribution = Counter(final_modes)
        
#         # Display results
#         print("🎯 CORE PERFORMANCE METRICS:")
#         core_metrics = ['hallucination_score', 'rouge1', 'rougeL', 'bleu_score', 'f1', 'length_score']
#         for metric in core_metrics:
#             avg_key = f'avg_{metric}'
#             std_key = f'std_{metric}'
#             if avg_key in avg_metrics:
#                 print(f"  {metric.replace('_', ' ').title()}: {avg_metrics[avg_key]:.3f} (±{avg_metrics[std_key]:.3f})")
        
#         print(f"\n⚡ WORKFLOW PERFORMANCE:")
#         print(f"  Average Processing Time: {avg_metrics.get('avg_processing_time', 0):.2f}s (±{avg_metrics.get('std_processing_time', 0):.2f})")
#         print(f"  Complete Answers: {complete_answers}/{len(all_metrics)} ({complete_answers/len(all_metrics)*100:.1f}%)")
#         print(f"  Mode Switches: {mode_switches}/{len(all_metrics)} ({mode_switches/len(all_metrics)*100:.1f}%)")
        
#         print(f"\n🔧 MODE DISTRIBUTION:")
#         for mode, count in mode_distribution.items():
#             print(f"  {mode}: {count}/{len(all_metrics)} ({count/len(all_metrics)*100:.1f}%)")
        
#         print(f"\n📏 RESPONSE ANALYSIS:")
#         print(f"  Avg Workflow Response Length: {avg_metrics.get('avg_workflow_response_length', 0):.1f} words")
#         print(f"  Avg Reference Response Length: {avg_metrics.get('avg_reference_response_length', 0):.1f} words")
        
#         # Performance interpretation
#         print("\n🏆 PERFORMANCE INTERPRETATION:")
#         halluc_score = avg_metrics.get('avg_hallucination_score', 0)
#         rouge1_score = avg_metrics.get('avg_rouge1', 0)
#         bleu_score = avg_metrics.get('avg_bleu_score', 0)
#         f1_score = avg_metrics.get('avg_f1', 0)
#         length_score = avg_metrics.get('avg_length_score', 0)
        
#         print(f"🔸 Hallucination Control: {'Excellent' if halluc_score > 0.7 else 'Good' if halluc_score > 0.5 else 'Needs Improvement'} ({halluc_score:.3f})")
#         print(f"🔸 Content Overlap (ROUGE-1): {'Excellent' if rouge1_score > 0.5 else 'Good' if rouge1_score > 0.3 else 'Needs Improvement'} ({rouge1_score:.3f})")
#         print(f"🔸 Precision Match (BLEU): {'Excellent' if bleu_score > 0.4 else 'Good' if bleu_score > 0.2 else 'Needs Improvement'} ({bleu_score:.3f})")
#         print(f"🔸 Overall F1: {'Excellent' if f1_score > 0.6 else 'Good' if f1_score > 0.4 else 'Needs Improvement'} ({f1_score:.3f})")
#         print(f"🔸 Length Consistency: {'Excellent' if length_score > 0.8 else 'Good' if length_score > 0.6 else 'Needs Improvement'} ({length_score:.3f})")
        
#         # Quality distribution
#         quality_grades = []
#         for m in all_metrics:
#             score = (m.get('rouge1', 0) + m.get('f1', 0) + m.get('hallucination_score', 0)) / 3
#             if score > 0.7:
#                 quality_grades.append('Excellent')
#             elif score > 0.5:
#                 quality_grades.append('Good')
#             else:
#                 quality_grades.append('Needs Improvement')
        
#         quality_dist = Counter(quality_grades)
#         print(f"\n📊 QUALITY DISTRIBUTION:")
#         for grade, count in quality_dist.items():
#             print(f"  {grade}: {count}/{len(all_metrics)} ({count/len(all_metrics)*100:.1f}%)")
        
#         print("\n📋 METRICS GUIDE:")
#         print("🔹 Higher scores = Better performance for all metrics")
#         print("🔹 Hallucination: 1.0 = No unsupported content")
#         print("🔹 ROUGE: Recall-oriented overlap with reference")
#         print("🔹 BLEU: Precision-oriented similarity")
#         print("🔹 F1: Balance of precision and recall")
#         print("🔹 Length Score: Response length similarity")
        
#         return {
#             'avg_metrics': avg_metrics,
#             'detailed_results': detailed_results,
#             'summary': {
#                 'total_samples': len(all_metrics),
#                 'complete_answers_rate': complete_answers/len(all_metrics),
#                 'mode_switches_rate': mode_switches/len(all_metrics),
#                 'quality_distribution': quality_dist,
#                 'mode_distribution': mode_distribution
#             }
#         }

# def main():
#     """Main evaluation function"""
#     # Configuration
#     dataset_path = r"C:\Users\Anushree\Desktop\Capital_One_Launchpad\Dataset\cleaned_natural_farming_dataset (1).csv"  # Update with your dataset path
#     num_samples = 10
#     initial_mode = "tooling"  # Can be changed to "tooling"
    
#     print("🌾 Agriculture Workflow Comprehensive Evaluation System")
#     print("=" * 80)
    
#     # Initialize evaluator
#     try:
#         evaluator = WorkflowEvaluator(dataset_path)
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         print("Please ensure the dataset path is correct and the file exists.")
#         return
    
#     # Run evaluation
#     try:
#         results = evaluator.run_evaluation(num_samples=num_samples, mode=initial_mode)
        
#         # Save detailed results to file
#         timestamp = int(time.time())
#         results_file = f"workflow_evaluation_results_{timestamp}.csv"
        
#         # Create DataFrame with detailed results
#         detailed_df = pd.DataFrame([
#             {
#                 'prompt': r['prompt'][:200] + "..." if len(r['prompt']) > 200 else r['prompt'],
#                 'workflow_response_length': r['metrics']['workflow_response_length'],
#                 'reference_response_length': r['metrics']['reference_response_length'],
#                 'hallucination_score': r['metrics']['hallucination_score'],
#                 'rouge1': r['metrics']['rouge1'],
#                 'rougeL': r['metrics']['rougeL'],
#                 'bleu_score': r['metrics']['bleu_score'],
#                 'f1': r['metrics']['f1'],
#                 'length_score': r['metrics']['length_score'],
#                 'processing_time': r['metrics']['processing_time'],
#                 'is_answer_complete': r['metrics']['is_answer_complete'],
#                 'final_mode': r['metrics']['final_mode'],
#                 'switched_modes': r['metrics']['switched_modes']
#             }
#             for r in results['detailed_results']
#         ])
        
#         detailed_df.to_csv(results_file, index=False)
#         print(f"\n💾 Detailed results saved to: {results_file}")
        
#     except Exception as e:
#         print(f"Error during evaluation: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import random
import time
from collections import Counter
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
from datasets import load_dataset

# Import your workflow
from workflow import run_workflow

# Download required NLTK data
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

class WorkflowEvaluator:
    def __init__(self):
        """Initialize evaluator with KisanVaani dataset"""
        print("Loading KisanVaani Agriculture Q&A English dataset...")
        self.dataset = load_dataset("KisanVaani/agriculture-qa-english-only")
        self.train_data = self.dataset["train"]
        print(f"Loaded dataset with {len(self.train_data)} samples")
        
        # Show sample data structure
        sample = self.train_data[0]
        print(f"Sample question: {sample['question']}")
        print(f"Sample answer: {sample['answers']}")
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if text is None or pd.isna(text):
            return ""
        
        text = str(text).strip()
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        return text
    
    def calculate_hallucination_score(self, candidate, reference, context=""):
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

    def calculate_rouge_scores(self, candidate, reference):
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

    def calculate_bleu_score(self, candidate, reference):
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

    def calculate_precision_recall_f1(self, candidate, reference):
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
    
    def calculate_response_length_score(self, candidate, reference):
        """Calculate length similarity score"""
        if not candidate or not reference:
            return 0.0
        
        candidate_len = len(candidate.split())
        reference_len = len(reference.split())
        
        if reference_len == 0:
            return 0.0
        
        # Length ratio score (closer to 1.0 is better)
        length_ratio = min(candidate_len, reference_len) / max(candidate_len, reference_len)
        return length_ratio
    
    def get_valid_samples(self, num_samples):
        """Get valid samples from the dataset"""
        valid_samples = []
        
        for sample in self.train_data:
            question = sample.get("question")
            answer = sample.get("answers")  # Note: 'answers' key in KisanVaani dataset
            
            if (question is not None and answer is not None and
                len(str(question).strip()) > 10 and
                len(str(answer).strip()) > 5):
                valid_samples.append(sample)
        
        if len(valid_samples) < num_samples:
            num_samples = len(valid_samples)
            print(f"Warning: Only {num_samples} valid samples available")
        
        return random.sample(valid_samples, num_samples)
    
    def evaluate_sample(self, question, reference_response, mode="tooling"):
        """Evaluate a single sample using the workflow"""
        print(f"Processing question: {question[:100]}...")
        
        # Get workflow response
        start_time = time.time()
        try:
            result = run_workflow(question, mode, image_path=None)
            workflow_response = result.get('answer', '')
            processing_time = time.time() - start_time
            
            # Extract additional workflow metrics
            is_answer_complete = result.get('is_answer_complete', False)
            final_mode = result.get('final_mode', mode)
            switched_modes = result.get('switched_modes', False)
            
        except Exception as e:
            print(f"Error running workflow: {e}")
            workflow_response = ""
            processing_time = 0.0
            is_answer_complete = False
            final_mode = mode
            switched_modes = False
        
        # Clean texts
        cleaned_workflow = self.clean_text(workflow_response)
        cleaned_reference = self.clean_text(reference_response)
        cleaned_question = self.clean_text(question)
        
        # Calculate all metrics
        metrics = {}
        
        # Overlap metrics
        metrics['hallucination_score'] = self.calculate_hallucination_score(
            cleaned_workflow, cleaned_reference, cleaned_question
        )
        
        rouge_scores = self.calculate_rouge_scores(cleaned_workflow, cleaned_reference)
        metrics.update(rouge_scores)
        
        metrics['bleu_score'] = self.calculate_bleu_score(cleaned_workflow, cleaned_reference)
        
        prf_scores = self.calculate_precision_recall_f1(cleaned_workflow, cleaned_reference)
        metrics.update(prf_scores)
        
        # Length similarity
        metrics['length_score'] = self.calculate_response_length_score(cleaned_workflow, cleaned_reference)
        
        # Workflow-specific metrics
        metrics['processing_time'] = processing_time
        metrics['is_answer_complete'] = is_answer_complete
        metrics['final_mode'] = final_mode
        metrics['switched_modes'] = switched_modes
        
        # Response lengths
        metrics['workflow_response_length'] = len(cleaned_workflow.split())
        metrics['reference_response_length'] = len(cleaned_reference.split())
        
        return {
            'metrics': metrics,
            'workflow_response': cleaned_workflow,
            'reference_response': cleaned_reference,
            'question': cleaned_question
        }
    
    def run_evaluation(self, num_samples=30, mode="tooling"):
        """Run comprehensive evaluation on random samples"""
        print(f"\n🚀 Agriculture Workflow Comprehensive Evaluation")
        print(f"🌾 Dataset: KisanVaani/agriculture-qa-english-only")
        print(f"📊 Evaluating {num_samples} random samples")
        print(f"🔧 Initial Mode: {mode}")
        print("=" * 80)
        
        # Get valid samples
        samples = self.get_valid_samples(num_samples)
        
        # Storage for results
        all_metrics = []
        detailed_results = []
        
        # Process each sample
        for idx, sample in enumerate(samples, 1):
            question = sample['question']
            reference = sample['answers']  # Note: 'answers' key in KisanVaani dataset
            
            print(f"\n{'='*60}")
            print(f"Sample {idx}/{len(samples)}")
            print(f"{'='*60}")
            
            result = self.evaluate_sample(question, reference, mode)
            
            all_metrics.append(result['metrics'])
            detailed_results.append(result)
            
            # Display sample metrics
            metrics = result['metrics']
            print(f"\n📊 Sample {idx} Metrics:")
            print(f"  🔍 Hallucination Score: {metrics['hallucination_score']:.3f}")
            print(f"  📝 ROUGE-1: {metrics['rouge1']:.3f}")
            print(f"  📝 ROUGE-L: {metrics['rougeL']:.3f}")
            print(f"  🎯 BLEU Score: {metrics['bleu_score']:.3f}")
            print(f"  ⚖️  F1 Score: {metrics['f1']:.3f}")
            print(f"  📏 Length Score: {metrics['length_score']:.3f}")
            print(f"  ⏱️  Processing Time: {metrics['processing_time']:.2f}s")
            print(f"  ✅ Answer Complete: {metrics['is_answer_complete']}")
            print(f"  🔄 Final Mode: {metrics['final_mode']}")
            print(f"  🔀 Switched Modes: {metrics['switched_modes']}")
            
            print(f"\n📝 Response Preview:")
            print(f"  Question:  {result['question'][:100]}...")
            print(f"  Reference: {result['reference_response'][:150]}...")
            print(f"  Workflow:  {result['workflow_response'][:150]}...")
        
        return self.calculate_final_results(all_metrics, detailed_results)
    
    def calculate_final_results(self, all_metrics, detailed_results):
        """Calculate and display final comprehensive results"""
        print("\n" + "=" * 80)
        print("📈 FINAL COMPREHENSIVE RESULTS - Agriculture Workflow Evaluation")
        print("🌾 Dataset: KisanVaani/agriculture-qa-english-only")
        print("=" * 80)
        
        # Calculate averages for numerical metrics
        numerical_metrics = [
            'hallucination_score', 'rouge1', 'rouge2', 'rougeL', 'bleu_score',
            'precision', 'recall', 'f1', 'length_score', 'processing_time',
            'workflow_response_length', 'reference_response_length'
        ]
        
        avg_metrics = {}
        for metric in numerical_metrics:
            values = [m[metric] for m in all_metrics if metric in m]
            if values:
                avg_metrics[f'avg_{metric}'] = np.mean(values)
                avg_metrics[f'std_{metric}'] = np.std(values)
        
        # Calculate workflow-specific metrics
        complete_answers = sum(1 for m in all_metrics if m.get('is_answer_complete', False))
        mode_switches = sum(1 for m in all_metrics if m.get('switched_modes', False))
        
        final_modes = [m.get('final_mode', 'unknown') for m in all_metrics]
        mode_distribution = Counter(final_modes)
        
        # Display results
        print("🎯 CORE PERFORMANCE METRICS:")
        core_metrics = ['hallucination_score', 'rouge1', 'rougeL', 'bleu_score', 'f1', 'length_score']
        for metric in core_metrics:
            avg_key = f'avg_{metric}'
            std_key = f'std_{metric}'
            if avg_key in avg_metrics:
                print(f"  {metric.replace('_', ' ').title()}: {avg_metrics[avg_key]:.3f} (±{avg_metrics[std_key]:.3f})")
        
        print(f"\n⚡ WORKFLOW PERFORMANCE:")
        print(f"  Average Processing Time: {avg_metrics.get('avg_processing_time', 0):.2f}s (±{avg_metrics.get('std_processing_time', 0):.2f})")
        print(f"  Complete Answers: {complete_answers}/{len(all_metrics)} ({complete_answers/len(all_metrics)*100:.1f}%)")
        print(f"  Mode Switches: {mode_switches}/{len(all_metrics)} ({mode_switches/len(all_metrics)*100:.1f}%)")
        
        print(f"\n🔧 MODE DISTRIBUTION:")
        for mode, count in mode_distribution.items():
            print(f"  {mode}: {count}/{len(all_metrics)} ({count/len(all_metrics)*100:.1f}%)")
        
        print(f"\n📏 RESPONSE ANALYSIS:")
        print(f"  Avg Workflow Response Length: {avg_metrics.get('avg_workflow_response_length', 0):.1f} words")
        print(f"  Avg Reference Response Length: {avg_metrics.get('avg_reference_response_length', 0):.1f} words")
        
        # Performance interpretation
        print("\n🏆 PERFORMANCE INTERPRETATION:")
        halluc_score = avg_metrics.get('avg_hallucination_score', 0)
        rouge1_score = avg_metrics.get('avg_rouge1', 0)
        bleu_score = avg_metrics.get('avg_bleu_score', 0)
        f1_score = avg_metrics.get('avg_f1', 0)
        length_score = avg_metrics.get('avg_length_score', 0)
        
        print(f"🔸 Hallucination Control: {'Excellent' if halluc_score > 0.7 else 'Good' if halluc_score > 0.5 else 'Needs Improvement'} ({halluc_score:.3f})")
        print(f"🔸 Content Overlap (ROUGE-1): {'Excellent' if rouge1_score > 0.5 else 'Good' if rouge1_score > 0.3 else 'Needs Improvement'} ({rouge1_score:.3f})")
        print(f"🔸 Precision Match (BLEU): {'Excellent' if bleu_score > 0.4 else 'Good' if bleu_score > 0.2 else 'Needs Improvement'} ({bleu_score:.3f})")
        print(f"🔸 Overall F1: {'Excellent' if f1_score > 0.6 else 'Good' if f1_score > 0.4 else 'Needs Improvement'} ({f1_score:.3f})")
        print(f"🔸 Length Consistency: {'Excellent' if length_score > 0.8 else 'Good' if length_score > 0.6 else 'Needs Improvement'} ({length_score:.3f})")
        
        # Quality distribution
        quality_grades = []
        for m in all_metrics:
            score = (m.get('rouge1', 0) + m.get('f1', 0) + m.get('hallucination_score', 0)) / 3
            if score > 0.7:
                quality_grades.append('Excellent')
            elif score > 0.5:
                quality_grades.append('Good')
            else:
                quality_grades.append('Needs Improvement')
        
        quality_dist = Counter(quality_grades)
        print(f"\n📊 QUALITY DISTRIBUTION:")
        for grade, count in quality_dist.items():
            print(f"  {grade}: {count}/{len(all_metrics)} ({count/len(all_metrics)*100:.1f}%)")
        
        print("\n📋 METRICS GUIDE:")
        print("🔹 Higher scores = Better performance for all metrics")
        print("🔹 Hallucination: 1.0 = No unsupported content")
        print("🔹 ROUGE: Recall-oriented overlap with reference")
        print("🔹 BLEU: Precision-oriented similarity")
        print("🔹 F1: Balance of precision and recall")
        print("🔹 Length Score: Response length similarity")
        
        return {
            'avg_metrics': avg_metrics,
            'detailed_results': detailed_results,
            'summary': {
                'total_samples': len(all_metrics),
                'complete_answers_rate': complete_answers/len(all_metrics),
                'mode_switches_rate': mode_switches/len(all_metrics),
                'quality_distribution': quality_dist,
                'mode_distribution': mode_distribution
            }
        }

def main():
    """Main evaluation function"""
    # Configuration
    num_samples = 10  # Number of samples to evaluate
    initial_mode = "tooling"  # Can be changed to "reasoning"
    
    print("🌾 Agriculture Workflow Comprehensive Evaluation System")
    print("🌾 Using KisanVaani/agriculture-qa-english-only Dataset")
    print("=" * 80)
    
    # Initialize evaluator
    try:
        evaluator = WorkflowEvaluator()
    except Exception as e:
        print(f"Error loading KisanVaani dataset: {e}")
        print("Please ensure you have internet connection to download the dataset.")
        return
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(num_samples=num_samples, mode=initial_mode)
        
        # Save detailed results to file
        timestamp = int(time.time())
        results_file = f"workflow_evaluation_kisanvaani_results_{timestamp}.csv"
        
        # Create DataFrame with detailed results
        detailed_df = pd.DataFrame([
            {
                'question': r['question'][:200] + "..." if len(r['question']) > 200 else r['question'],
                'workflow_response_length': r['metrics']['workflow_response_length'],
                'reference_response_length': r['metrics']['reference_response_length'],
                'hallucination_score': r['metrics']['hallucination_score'],
                'rouge1': r['metrics']['rouge1'],
                'rougeL': r['metrics']['rougeL'],
                'bleu_score': r['metrics']['bleu_score'],
                'f1': r['metrics']['f1'],
                'length_score': r['metrics']['length_score'],
                'processing_time': r['metrics']['processing_time'],
                'is_answer_complete': r['metrics']['is_answer_complete'],
                'final_mode': r['metrics']['final_mode'],
                'switched_modes': r['metrics']['switched_modes']
            }
            for r in results['detailed_results']
        ])
        
        detailed_df.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        print(f"\n🔧 CONFIGURATION SUMMARY:")
        print(f"📊 Dataset: KisanVaani/agriculture-qa-english-only")
        print(f"🔍 Total Dataset Size: {len(evaluator.train_data)}")
        print(f"📝 Samples Evaluated: {num_samples}")
        print(f"⚙️  Initial Mode: {initial_mode}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()