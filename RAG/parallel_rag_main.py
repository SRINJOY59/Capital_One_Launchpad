import os
import sys
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any
import time
import hashlib

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

cache_base_dir = os.path.join(current_dir, "parallel_cache")
os.makedirs(cache_base_dir, exist_ok=True)

from dotenv import load_dotenv
from .workflow import Workflow
from agno.agent import Agent
from agno.models.google import Gemini
from .document_scorer import FastQuerySummaryScorer
from pydantic import BaseModel  



load_dotenv()

class GeneralQuestion(BaseModel):
    agriculture_related : bool
    generation : str

class ParallelRAGSystem:
    def __init__(self, model="gemini-2.0-flash", k=3):
        self.model = model
        self.k = k
        self.api_key = os.getenv("GOOGLE_API_KEY")  
        self.data_dir = Path(current_dir) / "Data"
        self.cache_base = Path(cache_base_dir)
        self.document_scorer = FastQuerySummaryScorer()
        
        self.query_router = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            show_tool_calls=False,
            markdown=True,
            response_model=GeneralQuestion,
            instructions="""You are a query classification agent. Your task is to determine if a user's question is related to agriculture and provide appropriate responses.

**Classification Rules:**
- Set agriculture_related to True if the question is about: farming, crops, soil, livestock, agricultural practices, plant diseases, fertilizers, irrigation, agricultural technology, farm management, agricultural economics, or any farming-related topic
- Set agriculture_related to False for all other topics

**Response Generation:**
- If agriculture_related is True: Set generation to "ROUTE_TO_RAG"
- If agriculture_related is False:
  - For greetings/casual conversation: Provide a friendly response and mention you specialize in agricultural topics
  - For non-agricultural questions: Politely explain that you can only help with agriculture-related questions and suggest they ask about farming topics instead

Always be helpful and polite in your responses."""
        )
        
        self.synthesizer = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            show_tool_calls=False,
            markdown=True,
            instructions="""You are a senior agricultural consultant with decades of field experience and research expertise. Your role is to provide comprehensive, practical advice to farmers, agricultural professionals, and stakeholders.

When responding to questions, draw upon your extensive knowledge to provide clear, actionable guidance that reflects real-world agricultural practices and scientific understanding.

**Response Style:**
- Write in a natural, conversational tone as if speaking directly to the questioner
- Provide practical, implementable advice based on proven agricultural methods
- Include specific recommendations with clear reasoning
- Address potential challenges and provide solutions
- Use accessible language while maintaining technical accuracy
- Structure information logically from general concepts to specific applications

**Content Focus:**
- Emphasize practical applicability and real-world results
- Include considerations for different farming scales and contexts
- Mention timing, seasonal factors, and regional considerations where relevant
- Provide cost-effective solutions and alternatives
- Address sustainability and long-term soil health
- Include risk management and troubleshooting guidance

**Professional Standards:**
- Maintain authoritative expertise while being approachable
- Provide confident recommendations based on established practices
- Acknowledge limitations or variables when appropriate
- Focus on actionable outcomes and measurable improvements
- Ensure all advice aligns with modern sustainable agriculture principles

Respond as a trusted advisor who understands both the science and the practical realities of farming operations."""
        )
    
    def get_data_files(self) -> List[Path]:
        supported_extensions = ['.csv', '.pdf']
        data_files = []
        
        if not self.data_dir.exists():
            print(f"Data directory not found: {self.data_dir}")
            return data_files
        
        search_locations = [
            self.data_dir / "CSV",
        ]
        
        print(f"Searching for files in:")
        for location in search_locations:
            exists = "✓" if location.exists() else "✗"
            if location.exists():
                pdf_count = len(list(location.glob("*.pdf")))
                csv_count = len(list(location.glob("*.csv")))
                print(f"   {exists} {location} ({pdf_count} PDFs, {csv_count} CSVs)")
            else:
                print(f"   {exists} {location}")
        
        for location in search_locations:
            if location.exists():
                for ext in supported_extensions:
                    for file_path in location.glob(f"*{ext}"):
                        if file_path.is_file() and file_path not in data_files:
                            data_files.append(file_path)
        
        return data_files
    
    def get_file_preview(self, file_path: Path) -> str:
        try:
            if file_path.suffix.lower() == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path, nrows=5)
                content = df.to_string()
                return content[:500]
            else:
                return file_path.name
        except:
            return file_path.name
    
    def score_and_select_files(self, question: str, data_files: List[Path], top_k: int = 2) -> List[Path]:
        print(f"Scoring {len(data_files)} files against query...")
        
        file_summaries = []
        for file_path in data_files:
            preview = self.get_file_preview(file_path)
            file_summaries.append(preview)
        
        scores = self.document_scorer.batch_score_summaries(question, file_summaries, "ultimate")
        
        top_indices = [idx for idx, score in scores[:top_k]]
        selected_files = [data_files[i] for i in top_indices]
        
        print(f"Selected top {len(selected_files)} files:")
        for i, (idx, score) in enumerate(scores[:top_k]):
            print(f"   {i+1}. Score: {score:.3f} - {data_files[idx].name}")
        
        return selected_files
    
    def get_file_cache_id(self, file_path: Path) -> str:
        file_info = f"{file_path.name}_{file_path.stat().st_size}_{file_path.stat().st_mtime}"
        return hashlib.md5(file_info.encode()).hexdigest()[:8]
    
    def setup_workflow_cache(self, file_path: Path) -> str:
        cache_id = self.get_file_cache_id(file_path)
        workflow_cache_dir = self.cache_base / f"workflow_{cache_id}"
        workflow_cache_dir.mkdir(exist_ok=True)
        return str(workflow_cache_dir)
    
    def run_single_workflow(self, file_path: Path, question: str) -> Dict[str, Any]:
        try:
            print(f"Processing: {file_path.name}")
            workflow_cache_dir = self.setup_workflow_cache(file_path)
            workflow = Workflow(
                model=self.model, 
                api_key=self.api_key, 
                k=self.k, 
                file_path=str(file_path),
                cache_dir=workflow_cache_dir  
            )
            inputs = {"question": question}
            start_time = time.time()
            result = workflow.run_workflow(inputs)
            end_time = time.time()
            print("response: ", result)
            response = result.get('generation', '')
            extractions = result.get('extractions', '') 

            return {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix,
                "success": True,
                "response": response,
                "processing_time": end_time - start_time,
                "workflow_type": result.get('workflow_type', 'standard'),
                "extractions": extractions,
                "cache_dir": workflow_cache_dir,
                "error": None
            }
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {str(e)}")
            return {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix,
                "success": False,
                "response": "",
                "processing_time": 0,
                "workflow_type": "failed",
                "extractions": "",
                "cache_dir": "",
                "error": str(e)
            }
    
    def run_parallel_workflows(self, question: str, selected_files: List[Path], max_workers: int = None) -> List[Dict[str, Any]]:
        if max_workers is None or max_workers > len(selected_files):
            max_workers = len(selected_files)
        
        print(f"\nStarting parallel processing with {max_workers} workers for {len(selected_files)} selected files...")
        
        results = []
        successful_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.run_single_workflow, file_path, question): file_path
                for file_path in selected_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["success"]:
                        successful_count += 1
                    
                    status = "✅" if result["success"] else "❌"
                    print(f"{status} Completed: {file_path.name} ({successful_count}/{len(selected_files)} processed)")
                        
                except Exception as e:
                    print(f"❌ Exception for {file_path.name}: {str(e)}")
                    results.append({
                        "file_name": file_path.name,
                        "success": False,
                        "error": str(e)
                    })
        
        print(f"✅ Processing completed with {successful_count} successful results out of {len(results)} processed files")
        return results
    
    def synthesize_results(self, question: str, workflow_results: List[Dict[str, Any]]) -> str:
        successful_results = [r for r in workflow_results if r["success"] and r["response"]]

        if not successful_results:
            return "I don't have sufficient information to answer this question comprehensively."

        synthesis_prompt = [
            f"Based on your extensive agricultural knowledge and expertise, provide a comprehensive answer to this question: {question}",
            "",
            "Available information from research and field data:",
            ""
        ]

        for i, result in enumerate(successful_results, 1):
            synthesis_prompt.append(f"Agricultural Information {i}:")
            synthesis_prompt.append(str(result['response']))
            if result['extractions']:
                synthesis_prompt.append(f"Key Points: {result['extractions']}")
            synthesis_prompt.append("---")

        synthesis_prompt.append("")
        synthesis_prompt.append(f'Drawing from this agricultural information and your professional expertise, provide a thorough answer to: "{question}"')
        synthesis_prompt.append("Present your response as if you are sharing your own knowledge and experience, without referencing any external sources or documents. Focus on practical guidance and actionable recommendations.")

        synthesis_prompt_str = "\n".join(synthesis_prompt)

        print("Synthesizing agricultural insights...")
        synthesized_response = self.synthesizer.run(synthesis_prompt_str)

        return synthesized_response.content
    
    def process_query(self, question: str, max_workers: int = None) -> Dict[str, Any]:
        print("=" * 80)
        print("🌾 PARALLEL RAG SYSTEM WITH DOCUMENT SCORING")
        print("=" * 80)
        
        start_time = time.time()
        
        router_result = self.query_router.run(question).content
        
        if not router_result.agriculture_related:
            end_time = time.time()
            total_time = end_time - start_time
            return {
                "question": question,
                "total_files_available": 0,
                "files_selected": 0,
                "total_files_processed": 0,
                "successful_workflows": 0,
                "failed_workflows": 0,
                "individual_results": [],
                "synthesized_answer": router_result.generation,
                "total_processing_time": total_time,
                "average_time_per_file": 0
            }
        
        all_data_files = self.get_data_files()
        if not all_data_files:
            return {"error": "No data files found"}

        selected_files = self.score_and_select_files(question, all_data_files, top_k = 2)

        workflow_results = self.run_parallel_workflows(question, selected_files, max_workers)
        
        synthesized_answer = self.synthesize_results(question, workflow_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        successful_count = sum(1 for r in workflow_results if r["success"])
        failed_count = len(workflow_results) - successful_count
        
        return {
            "question": question,
            "total_files_available": len(all_data_files),
            "files_selected": len(selected_files),
            "total_files_processed": len(workflow_results),
            "successful_workflows": successful_count,
            "failed_workflows": failed_count,
            "individual_results": workflow_results,
            "synthesized_answer": synthesized_answer,
            "total_processing_time": total_time,
            "average_time_per_file": total_time / len(workflow_results) if workflow_results else 0
        }
    
    def clear_all_caches(self):
        import shutil
        if self.cache_base.exists():
            shutil.rmtree(self.cache_base)
            print(f"🗑️ Cleared all parallel workflow caches")
    
    def get_cache_info(self) -> Dict[str, Any]:
        if not self.cache_base.exists():
            return {"total_caches": 0, "total_size_mb": 0}
        
        cache_dirs = list(self.cache_base.glob("workflow_*"))
        total_size = 0
        
        for cache_dir in cache_dirs:
            for file_path in cache_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        return {
            "total_caches": len(cache_dirs),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_directories": [d.name for d in cache_dirs]
        }

def main():
    parallel_rag = ParallelRAGSystem(
        model="gemini-2.0-flash", 
        k=3
    )
    
    cache_info = parallel_rag.get_cache_info()
    print(f"📁 Cache Info: {cache_info['total_caches']} caches, {cache_info['total_size_mb']:.2f} MB")
    
    test_questions = [
        "What are the main agricultural practices mentioned in the documents?",
        "Provide information about crop recommendations and soil requirements",
        "What research findings or methodologies are discussed?",
        "Summarize the key insights about farming techniques and technologies"
    ]
    
    question = test_questions[0]
    print(f"Processing Question: {question}")
    
    result = parallel_rag.process_query(question)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    cache_info_after = parallel_rag.get_cache_info()
    print(f"\n📁 Cache Info After: {cache_info_after['total_caches']} caches, {cache_info_after['total_size_mb']:.2f} MB")
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Question: {result['question']}")
    
    if result.get('total_files_available', 0) == 0 and result.get('files_selected', 0) == 0:
        print("Non-agriculture query - Direct response:")
        print(result['synthesized_answer'])
        print(f"Total Time: {result['total_processing_time']:.2f}s")
        return
    
    print(f"Files Available: {result['total_files_available']}")
    print(f"Files Selected: {result['files_selected']}")
    print(f"Files Processed: {result['total_files_processed']}")
    print(f"Successful: {result['successful_workflows']}")
    print(f"Failed: {result['failed_workflows']}")
    print(f"Total Time: {result['total_processing_time']:.2f}s")
    print(f"Avg Time/File: {result['average_time_per_file']:.2f}s")
    
    print("\nSYNTHESIZED ANSWER:")
    print("-" * 60)
    print(result['synthesized_answer'])
    
    print("\nINDIVIDUAL WORKFLOW RESULTS:")
    print("-" * 60)
    for res in result['individual_results']:
        status = "✅" if res['success'] else "❌"
        print(f"{status} {res['file_name']}: {res['processing_time']:.2f}s")
        if not res['success']:
            print(f"   Error: {res['error']}")

if __name__ == "__main__":
    main()