from workflow import run_workflow

def run_manual_tests():
    """Run the original manual test questions"""
    import time
    questions = [
        "Estimate crop yield for wheat in Punjab in winter of 2025",
        "How can farmers manage pest outbreaks in cotton fields?",
        "What is the market price trend for wheat in India?",
        "How to prevent fungal diseases in tomato crops?",
    ]
    image_queries = [
        ("Analyze this crop disease", "Images/Crop/crop_disease.jpg"),
        ("Check for pests in this image", "Images/Pests/jpg_0.jpg"),
    ]

    mode = input("Select initial mode (rag/tooling): ").strip().lower()
    if mode not in ["rag", "tooling"]:
        print("Invalid mode. Using 'rag' as default.")
        mode = "rag"

    query_type = input("Test type (text/image): ").strip().lower()
    if query_type == "image":
        test_queries = image_queries
    else:
        test_queries = [(q, None) for q in questions]

    for idx, (user_query, image_path) in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Question {idx}: {user_query}")
        if image_path:
            print(f"Image Path: {image_path}")
        print(f"Initial Mode: {mode.upper()}")
        print('='*80)

        start_time = time.time()
        result = run_workflow(user_query, mode, image_path)
        end_time = time.time()

        print(f"Answer: {result['answer']}")
        print(f"\nQuality Metrics:")
        print(f"  - Is Answer Complete: {result['is_answer_complete']}")
        print(f"  - Final Mode: {result['final_mode']}")
        print(f"  - Switched Modes: {result['switched_modes']}")
        print(f"  - Is Image Query: {result['is_image_query']}")
        print(f"  - Processing Time: {end_time - start_time:.2f}s")

        if result['answer_quality_grade'].get('reasoning'):
            print(f"  - Quality Grade Reasoning: {result['answer_quality_grade']['reasoning']}")

def run_mcq_evaluation():
    """Run MCQ dataset evaluation"""
    from mcq_evaluation import MCQEvaluator
    import pandas as pd
    import time
    
    evaluator = MCQEvaluator()
    
    # Load dataset
    print("Loading MCQ dataset...")
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
    print(f"\nStarting MCQ evaluation with mode: {mode}")
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

def main():
    print("Select evaluation mode:")
    print("1. Manual Test Questions (original)")
    print("2. MCQ Dataset Evaluation")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        run_manual_tests()
    elif choice == "2":
        run_mcq_evaluation()
    else:
        print("Invalid choice. Running manual tests by default.")
        run_manual_tests()

if __name__ == "__main__":
    main()