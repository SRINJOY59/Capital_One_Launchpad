import pandas as pd
import time
from datasets import load_dataset
from workflow import run_workflow
from verifier import Verifier
import json
from typing import Dict, List, Tuple
import re

class MCQEvaluator:
    def __init__(self):
        """Initialize the MCQ evaluator with verifier agent"""
        self.verifier = Verifier()
        self.results = []
        
    def load_dataset(self, dataset_name: str = "bharatgenai/BhashaBench-Krishi", 
                    config: str = "English", split: str = "test"):
        """Load the MCQ dataset from HuggingFace"""
        try:
            dataset = load_dataset(dataset_name, config, split=split)
            return dataset.to_pandas()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def extract_option_from_answer(self, answer: str) -> str:
        """Extract option (A, B, C, D) from the workflow answer"""
        # Look for patterns like "Option A", "A)", "A.", "(A)", "Answer: A", etc.
        patterns = [
            r'\b([ABCD])\)',  # A), B), etc.
            r'\b([ABCD])\.',  # A., B., etc.
            r'\(([ABCD])\)',  # (A), (B), etc.
            r'Option\s+([ABCD])',  # Option A, Option B, etc.
            r'Answer\s*:?\s*([ABCD])',  # Answer: A, Answer A, etc.
            r'correct\s+answer\s+is\s+([ABCD])',  # correct answer is A, etc.
            r'\b([ABCD])\b(?=\s|$)'  # Standalone A, B, C, D
        ]
        
        answer_upper = answer.upper()
        
        for pattern in patterns:
            match = re.search(pattern, answer_upper, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return "UNKNOWN"
    
    def format_mcq_question(self, row: pd.Series) -> str:
        """Format the MCQ question with options for the workflow"""
        question = f"""Question: {row['question']}

A) {row['option_a']}
B) {row['option_b']}
C) {row['option_c']}
D) {row['option_d']}

Please provide your answer by selecting one of the options (A, B, C, or D) and explain your reasoning."""
        
        return question
    
    def evaluate_single_question(self, row: pd.Series, mode: str = "rag") -> Dict:
        """Evaluate a single MCQ question"""
        print(f"\nEvaluating Question ID: {row['id']}")
        print(f"Question: {row['question'][:100]}...")
        
        # Format question for workflow
        formatted_question = self.format_mcq_question(row)
        
        # Get workflow answer
        start_time = time.time()
        try:
            workflow_result = run_workflow(formatted_question, mode, None)
            processing_time = time.time() - start_time
            workflow_answer = workflow_result.get('answer', '')
        except Exception as e:
            print(f"Error in workflow: {e}")
            workflow_answer = ""
            processing_time = 0
            workflow_result = {}
        
        # Extract predicted option
        predicted_option = self.extract_option_from_answer(workflow_answer)
        ground_truth = row['correct_answer'].upper()
        
        print(f"Predicted: {predicted_option}, Ground Truth: {ground_truth}")
        
        # Verify using verifier agent
        verification_result = None
        try:
            verification_result = self.verifier.verify(
                question=row['question'],
                model_answer=predicted_option,
                ground_truth=ground_truth
            )
        except Exception as e:
            print(f"Error in verification: {e}")
        
        # Calculate accuracy
        is_correct = (predicted_option == ground_truth) and (predicted_option != "UNKNOWN")
        
        result = {
            'id': row['id'],
            'question': row['question'],
            'ground_truth': ground_truth,
            'predicted_option': predicted_option,
            'workflow_answer': workflow_answer,
            'is_correct': is_correct,
            'processing_time': processing_time,
            'language': row.get('language', 'unknown'),
            'subject_domain': row.get('subject_domain', 'unknown'),
            'question_level': row.get('question_level', 'unknown'),
            'question_type': row.get('question_type', 'MCQ'),
            'verification_result': verification_result,
            'workflow_metrics': {
                'is_answer_complete': workflow_result.get('is_answer_complete', False),
                'final_mode': workflow_result.get('final_mode', mode),
                'switched_modes': workflow_result.get('switched_modes', False),
                'answer_quality_grade': workflow_result.get('answer_quality_grade', {})
            }
        }
        
        return result
    
    def evaluate_dataset(self, dataset_df: pd.DataFrame, mode: str = "rag", 
                        max_samples: int = None, start_idx: int = 0) -> Dict:
        """Evaluate the entire dataset or a subset"""
        
        if max_samples:
            dataset_df = dataset_df.iloc[start_idx:start_idx + max_samples]
        
        print(f"Evaluating {len(dataset_df)} questions...")
        print(f"Mode: {mode}")
        print("="*80)
        
        results = []
        correct_predictions = 0
        total_questions = len(dataset_df)
        
        for idx, row in dataset_df.iterrows():
            try:
                result = self.evaluate_single_question(row, mode)
                results.append(result)
                
                if result['is_correct']:
                    correct_predictions += 1
                
                # Print progress
                current_accuracy = (correct_predictions / len(results)) * 100
                print(f"Progress: {len(results)}/{total_questions} | "
                      f"Current Accuracy: {current_accuracy:.2f}%")
                
            except Exception as e:
                print(f"Error evaluating question {row.get('id', idx)}: {e}")
                continue
        
        # Calculate final metrics
        accuracy = (correct_predictions / len(results)) * 100 if results else 0
        avg_processing_time = sum(r['processing_time'] for r in results) / len(results) if results else 0
        
        # Calculate metrics by categories
        metrics_by_level = self._calculate_metrics_by_category(results, 'question_level')
        metrics_by_domain = self._calculate_metrics_by_category(results, 'subject_domain')
        
        evaluation_summary = {
            'total_questions': len(results),
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'avg_processing_time': avg_processing_time,
            'metrics_by_level': metrics_by_level,
            'metrics_by_domain': metrics_by_domain,
            'mode_used': mode,
            'detailed_results': results
        }
        
        return evaluation_summary
    
    def _calculate_metrics_by_category(self, results: List[Dict], category: str) -> Dict:
        """Calculate accuracy metrics by category (level, domain, etc.)"""
        category_metrics = {}
        
        for result in results:
            cat_value = result.get(category, 'unknown')
            if cat_value not in category_metrics:
                category_metrics[cat_value] = {'correct': 0, 'total': 0}
            
            category_metrics[cat_value]['total'] += 1
            if result['is_correct']:
                category_metrics[cat_value]['correct'] += 1
        
        # Calculate accuracy for each category
        for cat_value in category_metrics:
            total = category_metrics[cat_value]['total']
            correct = category_metrics[cat_value]['correct']
            category_metrics[cat_value]['accuracy'] = (correct / total) * 100 if total > 0 else 0
        
        return category_metrics
    
    def save_results(self, evaluation_summary: Dict, filename: str = "mcq_evaluation_results.json"):
        """Save evaluation results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(evaluation_summary, f, indent=2, default=str)
        print(f"Results saved to {filename}")
    
    def print_summary(self, evaluation_summary: Dict):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Total Questions: {evaluation_summary['total_questions']}")
        print(f"Correct Predictions: {evaluation_summary['correct_predictions']}")
        print(f"Overall Accuracy: {evaluation_summary['accuracy']:.2f}%")
        print(f"Average Processing Time: {evaluation_summary['avg_processing_time']:.2f}s")
        print(f"Mode Used: {evaluation_summary['mode_used']}")
        
        print("\nAccuracy by Question Level:")
        for level, metrics in evaluation_summary['metrics_by_level'].items():
            print(f"  {level}: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")
        
        print("\nAccuracy by Subject Domain:")
        for domain, metrics in evaluation_summary['metrics_by_domain'].items():
            print(f"  {domain}: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")


def main():
    """Main evaluation function"""
    evaluator = MCQEvaluator()
    
    # Load dataset
    print("Loading dataset...")
    dataset_df = evaluator.load_dataset("bharatgenai/BhashaBench-Krishi")
    
    if dataset_df is None:
        print("Failed to load dataset")
        return
    
    print(f"Dataset loaded: {len(dataset_df)} questions")
    
    # Configuration
    mode = input("Select mode (rag/tooling): ").strip().lower()
    if mode not in ["rag", "tooling"]:
        print("Invalid mode. Using 'rag' as default.")
        mode = "rag"
    
    max_samples = input("Enter max samples to evaluate (or press Enter for all): ").strip()
    max_samples = int(max_samples) if max_samples.isdigit() else None
    
    start_idx = input("Enter start index (default 0): ").strip()
    start_idx = int(start_idx) if start_idx.isdigit() else 0
    
    # Run evaluation
    print(f"\nStarting evaluation with mode: {mode}")
    evaluation_summary = evaluator.evaluate_dataset(
        dataset_df, 
        mode=mode, 
        max_samples=max_samples, 
        start_idx=start_idx
    )
    
    # Print and save results
    evaluator.print_summary(evaluation_summary)
    
    # Save results
    filename = f"mcq_evaluation_{mode}_{int(time.time())}.json"
    evaluator.save_results(evaluation_summary, filename)
    
    # Also save a CSV with detailed results for analysis
    results_df = pd.DataFrame(evaluation_summary['detailed_results'])
    csv_filename = f"mcq_detailed_results_{mode}_{int(time.time())}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"Detailed results saved to {csv_filename}")


if __name__ == "__main__":
    main()