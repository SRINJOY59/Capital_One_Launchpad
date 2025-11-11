import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
cache_base_dir = os.path.join(current_dir, "cache")
os.makedirs(cache_base_dir, exist_ok=True)

from dotenv import load_dotenv
from .workflow import Workflow
import time
import hashlib
from gptcache import Cache
from langchain.globals import set_llm_cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache

load_dotenv()


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

  
def main():
    model = "gemini-2.0-flash"
    k = 3
    file_path = r"Data/CSV/agriculture_irrigation_qa_chunk_01.csv"
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    workflow = Workflow(model, api_key, k, file_path)

    start = time.time()

    question = "Why is the location of the Indian Agricultural Research Institute important?"
    inputs = {"question": question}
    model_output = workflow.run_workflow(inputs)
    
    generated_text = model_output['generation']
        
    end = time.time()
    print("Time taken : ", end - start)
        
    print("Pipeline response: \n" , generated_text)
    
    
    
if __name__ == "__main__":
    main()
