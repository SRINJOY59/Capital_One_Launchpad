import pandas as pd
import random
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cache_base_dir = os.path.join(current_dir, "cache")
os.makedirs(cache_base_dir, exist_ok=True)

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
    cache_dir = os.path.join(cache_base_dir, "llm_cache", f"map_cache_{hashed_llm}")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=cache_dir),
    )


set_llm_cache(GPTCache(init_gptcache))


def llm_eval(response, query, reference, documents, model="llama-3.3-70b-versatile"):
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY must be set in the .env file.")

    llm = ChatGroq(
        model=model,
        api_key=groq_api_key,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    evaluator_llm = LangchainLLMWrapper(llm)

    if isinstance(documents, list):
        retrieved_contexts = [str(doc) for doc in documents]
    else:
        retrieved_contexts = [str(documents)]

    dataset = [{
        "user_input": str(query),
        "response": str(response),           
        "reference": str(reference),
        "retrieved_contexts": retrieved_contexts      
    }]
    
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    try:
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[LLMContextRecall(), Faithfulness()],
            llm=evaluator_llm
        )
        
        return {
            'context_recall': float(result['context_recall'].iloc[0]) if 'context_recall' in result else 0.0,
            'faithfulness': float(result['faithfulness'].iloc[0]) if 'faithfulness' in result else 0.0
        }
    except Exception as e:
        print(f"RAGAS evaluation error: {e}")
        return {'context_recall': 0.0, 'faithfulness': 0.0}