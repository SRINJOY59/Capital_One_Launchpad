import os
import sys
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dotenv import load_dotenv
from Internet_checker import Workflow
from adaptive_rag_class import LoadDocuments
from rouge import Rouge
from bert_score import score as bert_score  
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # For BLEU score (`pip install nltk`)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from ragas_evaluation import llm_eval
load_dotenv()

import hashlib
from gptcache import Cache
from langchain.globals import set_llm_cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )


set_llm_cache(GPTCache(init_gptcache))

import pandas as pd
import random

def evaluate_questions(questions_data, workflow, llm, num_questions=20):
    if len(questions_data) > num_questions:
        sampled_data = questions_data.sample(num_questions, random_state=42)
    else:
        sampled_data = questions_data

    results = []

    for idx, row in sampled_data.iterrows():
        question = str(row.get('question', row.get('query', '')))
        ground_truth = str(row.get('answer', row.get('ground_truth', row.get('response', ''))))
        
        if not question or not ground_truth:
            print(f"Skipping row {idx}: Missing question or ground truth")
            continue
        
        inputs = {"question": question}
        model_output = workflow.run_workflow(inputs)
        generated_text = model_output['generation']
        documents = model_output['documents']
        
        retrieved_contents = []
        for doc in documents[:2]:  # Use top 2 documents
            if hasattr(doc, 'page_content'):
                retrieved_contents.append(doc.page_content)
            elif isinstance(doc, str):
                retrieved_contents.append(doc)

        if model_output.get('extracted_info'):
            final_response = f"{generated_text}\n\nAdditional Info: {model_output['extracted_info']}"
        else:
            final_response = generated_text
        
        ragas_evaluation_result = llm_eval(final_response, question, ground_truth, retrieved_contents)
        
        rouge_l, bert_f1, cos_sim, bleu = calculate_evaluation_metrics(ground_truth, final_response)

        print(f"Question {idx}: RAGAS evaluation completed")
        
        results.append({
            'question': question,
            'ground_truth': ground_truth,
            'generated_response': final_response,
            'rouge_l': rouge_l,
            'bert_f1': bert_f1,
            'cosine_similarity': cos_sim,
            'bleu': bleu,
            'context_recall': ragas_evaluation_result.get('context_recall', 0),
            'faithfulness': ragas_evaluation_result.get('faithfulness', 0),
            'answer_relevancy': ragas_evaluation_result.get('answer_relevancy', 0),
            'context_precision': ragas_evaluation_result.get('context_precision', 0),
        })

    return results

def calculate_aggregate_metrics(results):
    if not results:
        return {}
    
    metrics = {
        'avg_rouge_l': sum(r['rouge_l'] for r in results) / len(results),
        'avg_bert_f1': sum(r['bert_f1'] for r in results) / len(results),
        'avg_cosine_similarity': sum(r['cosine_similarity'] for r in results) / len(results),
        'avg_bleu': sum(r['bleu'] for r in results) / len(results),
        'avg_context_recall': sum(r['context_recall'] for r in results) / len(results),
        'avg_faithfulness': sum(r['faithfulness'] for r in results) / len(results),
        'avg_answer_relevancy': sum(r['answer_relevancy'] for r in results) / len(results),
        'avg_context_precision': sum(r['context_precision'] for r in results) / len(results),
    }
    
    return metrics

def calculate_evaluation_metrics(ground_truth, model_output):

    rouge = Rouge()
    sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    smoothing_function = SmoothingFunction().method4

    rouge_scores = rouge.get_scores(model_output, ground_truth)
    rouge_l_score = rouge_scores[0]['rouge-l']['f']

    P, R, F1 = bert_score([model_output], [ground_truth], lang="en")
    bert_f1 = F1.mean().item()

    embeddings = sentence_transformer.encode([model_output, ground_truth])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    reference = ground_truth.split()
    candidate = model_output.split()
    bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing_function)

    print(f"Ground Truth: {ground_truth}")
    print(f"Model Output: {model_output}")
    print(f"ROUGE-L Score: {rouge_l_score:.4f}")
    print(f"BERTScore (F1): {bert_f1:.4f}")
    print(f"Cosine Similarity: {cos_sim:.4f}")
    print(f"BLEU Score: {bleu:.4f}\n")

    return rouge_l_score, bert_f1, cos_sim, bleu


def main():
    model = "llama-3.3-70b-versatile"
    k = 3
    data_path = r'your_dataset.csv'  
    
    api_key = os.getenv("GROQ_API_KEY")
    
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        api_key=api_key
    )
    
    workflow = Workflow(model, api_key, k, data_path)
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        print("Unsupported file format for evaluation. Please use CSV or JSON.")
        return

    results = evaluate_questions(df, workflow=workflow, llm=llm, num_questions=50)
    
    aggregate_metrics = calculate_aggregate_metrics(results)
    
    print("\n=== Evaluation Results ===")
    for metric, value in aggregate_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv("generic_rag_evaluation_results.csv", index=False)
    
    # Save aggregate metrics
    with open("aggregate_metrics.json", "w") as f:
        json.dump(aggregate_metrics, f, indent=2)
    
    print("\nEvaluation complete.")
    print("Detailed results saved to: generic_rag_evaluation_results.csv")
    print("Aggregate metrics saved to: aggregate_metrics.json")

if __name__ == "__main__":
    main()