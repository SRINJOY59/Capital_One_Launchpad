import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

cache_base_dir = os.path.join(current_dir, "cache")
os.makedirs(cache_base_dir, exist_ok=True)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from agno.embedder.google import GeminiEmbedder
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain import hub
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from .Agents.search_agent import Search
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from .Agents.answer_grader_agent import AnswerGrader
from .Agents.hallucinator_agent import HallucinationGrader
from .Agents.grader_agent import Grader
from .Agents.question_rewriter import QuestionRewriter
from .Agents.abstractor_agent import Abstractor
from pprint import pprint
from langgraph.graph import END
from .Agents.reflectionAgent import IntrospectiveAgent
# Additional imports for document loading
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
import json
import pandas as pd
import pathlib
import hashlib
import pickle
from gptcache import Cache
from langchain.globals import set_llm_cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache
import re
import math
import numpy as np
from collections import Counter

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


class LoadDocuments:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_extension = pathlib.Path(file_path).suffix.lower()

    def load_documents(self):
        """Load documents based on file extension"""
        if self.file_extension == '.csv':
            return self._load_csv()
        elif self.file_extension == '.pdf':
            return self._load_pdf()
        elif self.file_extension in ['.txt', '.md']:
            return self._load_text()
        elif self.file_extension in ['.docx', '.doc']:
            return self._load_word()
        elif self.file_extension == '.json':
            return self._load_json()
        else:
            raise ValueError(f"Unsupported file format: {self.file_extension}")

    def _load_csv(self):
        """Load CSV files with general format handling"""
        df = pd.read_csv(self.file_path)
        documents = []
        questions = []
        translations = []

        for _, row in df.iterrows():
            content_parts = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    if isinstance(val, list):
                        val_str = ", ".join(str(v) for v in val)
                    else:
                        val_str = str(val)
                    content_parts.append(f"{col}: {val_str}")
            content = "\n".join(content_parts)
            documents.append(Document(
                page_content=content,
                metadata={"row_index": row.name, "source": self.file_path}
            ))
            questions.append("General information query")
            translations.append(content[:200] + "..." if len(content) > 200 else content)

        return documents, questions, translations

    def _load_pdf(self):
        """Load PDF files"""
        loader = PyPDFLoader(self.file_path)
        pages = loader.load()
        documents = []
        questions = []
        translations = []

        for i, page in enumerate(pages):
            documents.append(Document(
                page_content=page.page_content,
                metadata={"page": i + 1, "source": self.file_path}
            ))
            questions.append(f"Information from page {i + 1}")
            translations.append(page.page_content[:200] + "..." if len(page.page_content) > 200 else page.page_content)

        return documents, questions, translations

    def _load_text(self):
        """Load text/markdown files"""
        loader = TextLoader(self.file_path, encoding='utf-8')
        docs = loader.load()
        documents = []
        questions = []
        translations = []

        for doc in docs:
            # Split large text files into chunks
            content = doc.page_content
            if len(content) > 2000:
                chunks = [content[i:i+2000] for i in range(0, len(content), 1800)]
                for j, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"chunk": j + 1, "source": self.file_path}
                    ))
                    questions.append(f"Information from text chunk {j + 1}")
                    translations.append(chunk[:200] + "..." if len(chunk) > 200 else chunk)
            else:
                documents.append(Document(
                    page_content=content,
                    metadata={"source": self.file_path}
                ))
                questions.append("Text document information")
                translations.append(content[:200] + "..." if len(content) > 200 else content)

        return documents, questions, translations

    def _load_word(self):
        """Load Word documents"""
        try:
            loader = UnstructuredWordDocumentLoader(self.file_path)
            docs = loader.load()
            documents = []
            questions = []
            translations = []

            for doc in docs:
                documents.append(Document(
                    page_content=doc.page_content,
                    metadata={"source": self.file_path}
                ))
                questions.append("Word document information")
                translations.append(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)

            return documents, questions, translations
        except Exception as e:
            raise ValueError(f"Error loading Word document: {str(e)}")

    def _load_json(self):
        """Load JSON files"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        questions = []
        translations = []

        if isinstance(data, list):
            for i, item in enumerate(data):
                content = json.dumps(item, indent=2) if isinstance(item, dict) else str(item)
                documents.append(Document(
                    page_content=content,
                    metadata={"item": i + 1, "source": self.file_path}
                ))
                questions.append(f"JSON item {i + 1} information")
                translations.append(content[:200] + "..." if len(content) > 200 else content)
        elif isinstance(data, dict):
            content = json.dumps(data, indent=2)
            documents.append(Document(
                page_content=content,
                metadata={"source": self.file_path}
            ))
            questions.append("JSON document information")
            translations.append(content[:200] + "..." if len(content) > 200 else content)
        else:
            content = str(data)
            documents.append(Document(
                page_content=content,
                metadata={"source": self.file_path}
            ))
            questions.append("JSON data information")
            translations.append(content)

        return documents, questions, translations
    
class MyEmbeddings:
    def __init__(self, model: str = None):
        self.embedder = GeminiEmbedder()
        # Create embedding cache directory
        self.cache_dir = os.path.join(cache_base_dir, "embedding_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "embedding_cache.json")
        self._cache = self._load_cache()

    def _load_cache(self):
        """Load cache from file if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f)
        except Exception as e:
            print(f"Warning: Could not save embedding cache: {e}")

    def __call__(self, text):
        """Make the class callable for FAISS compatibility"""
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._cache:
                embeddings.append(self._cache[text_hash])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)
        
        # Batch process uncached texts
        if uncached_texts:
            # Process in smaller batches to avoid timeout
            batch_size = 5
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i+batch_size]
                for j, text in enumerate(batch):
                    embedding = self.embedder.get_embedding(text)
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    self._cache[text_hash] = embedding
                    embeddings[uncached_indices[i+j]] = embedding
        
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self._cache:
            return self._cache[query_hash]
        
        embedding = self.embedder.get_embedding(query)
        self._cache[query_hash] = embedding
        self._save_cache()  # Save cache after each update
        return embedding
    
        
class ADAPTIVE_RAG:
    def __init__(self, model, api_key, k, file_path, cache_dir=None):
        self.load_documents = LoadDocuments(file_path)
        self.chat_memory = []
        
        # Initialize instance variables first
        self.model = model
        self.api_key = api_key
        self.k = k
        
        # Use custom cache directory if provided
        if cache_dir:
            self.cache_base_dir = cache_dir
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.cache_base_dir = os.path.join(current_dir, "cache")
        
        os.makedirs(self.cache_base_dir, exist_ok=True)
        
        self.vectorstore_dir = os.path.join(self.cache_base_dir, "vectorstore", "faiss_db")
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        
        # Fix paths to match actual structure: faiss_index subdirectory exists
        self.faiss_index_path = os.path.join(self.vectorstore_dir, "faiss_index")
        self.faiss_pkl_path = os.path.join(self.vectorstore_dir, "faiss_vectorstore.pkl")
        self.bm25_index_path = os.path.join(self.vectorstore_dir, "bm25_retriever.pkl")
        self.doc_splits_path = os.path.join(self.vectorstore_dir, "doc_splits.pkl")
        
        if self._vectorstore_exists():
            print("Loading existing vectorstore...")
            self.embd = MyEmbeddings()
            self._load_existing_retrievers()
        else:
            print("Creating new vectorstore...")
            self.documents, self.questions, self.translations = self.load_documents.load_documents()
            self.embd = MyEmbeddings()
            self.text_splitter = SemanticChunker(self.embd)
            self.doc_splits = self.text_splitter.split_documents(self.documents)
            
            doc_texts = [doc.page_content for doc in self.doc_splits]
            
            self.bm25_retriever = BM25Retriever.from_texts(
                doc_texts, 
                metadatas=[doc.metadata for doc in self.doc_splits]
            )
            self.bm25_retriever.k = min(k, 3)
            
            self.faiss_vectorstore = FAISS.from_documents(
                documents=self.doc_splits,
                embedding=self.embd
            )
            self.faiss_retriever = self.faiss_vectorstore.as_retriever(search_kwargs={"k": min(k, 3)})
            
            self._save_retrievers()
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[0.5, 0.5]  
        )
        
        self.compressor = CohereRerank(model="rerank-english-v3.0")
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, 
            base_retriever=self.ensemble_retriever
        )
        
        # Use only Google Gemini models
        self.llm = ChatGoogleGenerativeAI(model=model, api_key=api_key)
        
        self.generator_prompt = """
        You are an AI assistant for question-answering tasks. Use the provided context along with the previous chat history to deliver a precise and concise response. If the information is insufficient or unclear, acknowledge that you don't know. Keep the answer brief (three sentences or less) while maintaining clarity and relevance.

        """
        self.human_prompt = """
            Inputs:

            Question: {question}
            Context: {context}
            Chat History: {chat_history}
            Output:
            Answer:
        """
        self.rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.generator_prompt), 
                ("human", self.human_prompt)
            ]
        )
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        self.recursion_limit = 2
        self.recursion_counter = 0

        self.introspective_agent = IntrospectiveAgent(
            model_id=self.model,
        )

    def _vectorstore_exists(self):
        # Check for the actual file structure: faiss_index/index.faiss
        faiss_index_file = os.path.join(self.vectorstore_dir, "faiss_index", "index.faiss")
        return (os.path.exists(faiss_index_file) or 
                os.path.exists(self.faiss_pkl_path)) and os.path.exists(self.bm25_index_path)
    
    def _save_retrievers(self):
        try:
            self.faiss_vectorstore.save_local(self.faiss_index_path)
            print("FAISS vectorstore saved using native method.")
            
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump(self.bm25_retriever, f)
            
            with open(self.doc_splits_path, 'wb') as f:
                pickle.dump(self.doc_splits, f)
                
            print("BM25 retriever and documents saved successfully.")
        except Exception as e:
            print(f"Warning: Could not save retrievers: {e}")
    
    def _load_existing_retrievers(self):
        try:
            # Check for index.faiss in the faiss_index subdirectory
            faiss_index_file = os.path.join(self.vectorstore_dir, "faiss_index", "index.faiss")
            if os.path.exists(faiss_index_file):
                self.faiss_vectorstore = FAISS.load_local(
                    self.faiss_index_path, 
                    self.embd,
                    allow_dangerous_deserialization=True
                )
                print("FAISS vectorstore loaded using native method.")
            elif os.path.exists(self.faiss_pkl_path):
                with open(self.faiss_pkl_path, 'rb') as f:
                    self.faiss_vectorstore = pickle.load(f)
                print("FAISS vectorstore loaded from pickle backup.")
            else:
                raise FileNotFoundError("No FAISS vectorstore found")
                
            self.faiss_retriever = self.faiss_vectorstore.as_retriever(search_kwargs={"k": min(self.k, 3)})
            
            with open(self.bm25_index_path, 'rb') as f:
                self.bm25_retriever = pickle.load(f)
            
            if os.path.exists(self.doc_splits_path):
                with open(self.doc_splits_path, 'rb') as f:
                    self.doc_splits = pickle.load(f)
            
            print("Existing retrievers loaded successfully.")
        except Exception as e:
            print(f"Error loading existing retrievers: {e}")
            self.documents, self.questions, self.translations = self.load_documents.load_documents()
            self.text_splitter = SemanticChunker(self.embd)
            self.doc_splits = self.text_splitter.split_documents(self.documents)
            
            doc_texts = [doc.page_content for doc in self.doc_splits]
            
            self.bm25_retriever = BM25Retriever.from_texts(
                doc_texts, 
                metadatas=[doc.metadata for doc in self.doc_splits]
            )
            self.bm25_retriever.k = min(self.k, 3)
            
            self.faiss_vectorstore = FAISS.from_documents(
                documents=self.doc_splits,
                embedding=self.embd
            )
            self.faiss_retriever = self.faiss_vectorstore.as_retriever(search_kwargs={"k": min(self.k, 3)})
            
            self._save_retrievers()
    
    def save_vectorstore_manually(self):
        self._save_retrievers()
        
    def clear_vectorstore_cache(self):
        import shutil
        if os.path.exists(self.vectorstore_dir):
            shutil.rmtree(self.vectorstore_dir)
            print("Vectorstore cache cleared.")
        else:
            print("No vectorstore cache to clear.")
            
    def get_vectorstore_info(self):
        faiss_index_file = os.path.join(self.vectorstore_dir, "faiss_index", "index.faiss")
        info = {
            "faiss_exists": os.path.exists(faiss_index_file) or os.path.exists(self.faiss_pkl_path),
            "bm25_exists": os.path.exists(self.bm25_index_path),
            "doc_splits_exists": os.path.exists(self.doc_splits_path),
            "vectorstore_dir": self.vectorstore_dir,
            "num_documents": len(self.doc_splits) if hasattr(self, 'doc_splits') else 0
        }
        return info
        
    def retrieve(self, state):
        question = state["question"]
        documents = self.compression_retriever.invoke(question)   
        return {"documents": documents, "question": question}
    def abstraction(self, state):
        content = state["documents"]
        abstractor = Abstractor(self.model)
        extractions = abstractor.abstract(content)
        print("extracted info: ", extractions)
        return {"documents": content, "question": state["question"], "extractions": extractions}
    def generate(self, state):
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"context": documents, "question": question, "chat_history": self.chat_memory})
        if len(self.chat_memory) < 5:
            self.chat_memory.append(generation)
        else:
            self.chat_memory.pop(0)
            self.chat_memory.append(generation)
        if state.get("extractions") is not None:
            return {"documents": documents, "question": question,"extractions": state["extractions"], "generation": generation}
        return {"documents": documents, "question": question, "generation": generation}
        
    def fast_generate(self, state):
        question = state["question"]
        documents = state["documents"]
        
        fast_prompt = f"Context: {documents}\n\nQuestion: {question}\n\nAnswer:"
        generation = self.llm.invoke(fast_prompt).content
        
        if len(self.chat_memory) < 5:
            self.chat_memory.append(generation)
        else:
            self.chat_memory.pop(0)
            self.chat_memory.append(generation)
            
        return {"documents": documents, "question": question, "generation": generation}
        
    def grade_documents(self, state):
        question = state["question"]
        print("quest")
        documents = state["documents"]
        filtered_docs = []
        
        top_docs = documents[:3] if len(documents) > 3 else documents
        
        for d in top_docs:
            if len(filtered_docs) >= 2:  # Early stopping - we have enough good docs
                break
            grader = Grader(self.model)
            grade = grader.grade_documents(question, d.page_content)
            if grade == "yes":
                filtered_docs.append(d)
                
        return {"documents": filtered_docs, "question": question}
        
    def transform_query(self, state):
        question = state["question"]
        documents = state["documents"]
        questionRewriter = QuestionRewriter(self.model)
        better_question = questionRewriter.re_write_question(question)
        return {"documents": documents, "question": better_question}
        
    def web_search(self, state):
        question = state["question"]
        search = Search(self.k)
        docs = search.web_search(question)
        return {"documents": docs, "question": question}
        
    def decide_to_generate(self, state):
        question = state["question"]
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        if not filtered_documents:
            return "transform_query"
        else:
            return "generate"
            
    def grade_generation_v_documents_and_question(self, state):
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        print("---CHECK HALLUCINATIONS---")
        hallucinationGrader = HallucinationGrader(self.model)
        grade = hallucinationGrader.grade_hallucinations(documents, generation)
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            answerGrader = AnswerGrader(self.model)
            grade = answerGrader.grade_answer(question, generation)
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        
    def introspective_agent_response(self, state):
        question = state["question"]
        extracted_info = state.get("extractions")
        retrieved_docs = state["documents"]
        
        final_prompt = f"""
            This is the question given by the user : {question}
            These are the most relevant retrieved documents : {retrieved_docs[0]} 
        """
        response = self.introspective_agent.introspect_and_respond(final_prompt)
        return {"generation": str(response), "question": question, "extractions": extracted_info, "documents": retrieved_docs}
        
    def track_recursion_and_retrieve(self, state):
        """
        Track the number of recursions when going to 'transform_query'.
        End the workflow if the recursion limit (2) is reached.
        """
        if not hasattr(self, "recursion_counter"):
            self.recursion_counter = 0
        if self.recursion_counter < 2:
            self.recursion_counter += 1
            return "retrieve"
        else:
            return "end_due_to_limit"

    def end_due_to_limit(self, state):
        """
        End the workflow because the recursion limit was reached.
        """
        print("Recursion limit reached. Ending workflow.")
        return END

    # === COMPLEXITY ANALYSIS METHODS ===
    
    def analyze_query_complexity(self, state):
        question = state["question"]
        documents = state["documents"]
        
        length_score = self._calculate_length_complexity(question)
        linguistic_score = self._calculate_linguistic_complexity(question)
        semantic_score = self._calculate_semantic_complexity(question)
        structure_score = self._calculate_structural_complexity(question)
        domain_score = self._calculate_domain_complexity(question)
        intent_score = self._calculate_intent_complexity(question)
        
        doc_quality_score = self._calculate_document_quality(question, documents)
        
        complexity_vector = np.array([length_score, linguistic_score, semantic_score, 
                                    structure_score, domain_score, intent_score])
        weights = np.array([0.1, 0.25, 0.2, 0.15, 0.15, 0.15])
        
        total_complexity = np.dot(complexity_vector, weights)
        normalized_complexity = 1 / (1 + math.exp(-2 * (total_complexity - 2.5)))
        
        print(f"Complexity scores: {complexity_vector}")
        print(f"Total complexity: {total_complexity:.3f}")
        print(f"Normalized complexity: {normalized_complexity:.3f}")
        print(f"Document quality: {doc_quality_score:.3f}")
        
        return self._route_decision(normalized_complexity, doc_quality_score, question)
    
    def _calculate_length_complexity(self, question):
        words = question.split()
        word_count = len(words)
        char_count = len(question)
        
        length_factor = math.log(1 + word_count) / math.log(20)
        avg_word_length = char_count / max(word_count, 1)
        complexity_factor = math.tanh((avg_word_length - 4) / 2)
        
        return min(length_factor + complexity_factor, 1.0)
    
    def _calculate_structural_complexity(self, question):
        clause_indicators = [',', ';', ':', ' and ', ' or ', ' but ', ' however ', ' therefore ']
        clause_density = sum(question.lower().count(indicator) for indicator in clause_indicators) / len(question)
        
        question_marks = question.count('?')
        parentheses = (question.count('(') + question.count(')')) / 2
        quotes = (question.count('"') + question.count("'")) / 2
        
        structural_features = np.array([clause_density * 100, question_marks, parentheses, quotes])
        feature_weights = np.array([0.4, 0.3, 0.15, 0.15])
        
        weighted_score = np.dot(structural_features, feature_weights)
        return math.tanh(weighted_score / 2)
    
    def _calculate_linguistic_complexity(self, question):
        question_lower = question.lower()
        
        high_complexity_patterns = [
            r'\bexplain\b', r'\banalyze\b', r'\bevaluate\b', r'\bassess\b',
            r'\bdiscuss\b', r'\bexamine\b', r'\bcritique\b', r'\bjustify\b',
            r'\bdemonstrate\b', r'\bsynthesize\b', r'\binterpret\b'
        ]
        
        medium_complexity_patterns = [
            r'\bhow\b', r'\bwhy\b', r'\bcompare\b', r'\bcontrast\b',
            r'\bdescribe\b', r'\bidentify\b', r'\bsummarize\b'
        ]
        
        simple_patterns = [
            r'\bwhat\b', r'\bwhen\b', r'\bwhere\b', r'\bwho\b', r'\bwhich\b'
        ]
        
        high_matches = sum(1 for pattern in high_complexity_patterns if re.search(pattern, question_lower))
        medium_matches = sum(1 for pattern in medium_complexity_patterns if re.search(pattern, question_lower))
        simple_matches = sum(1 for pattern in simple_patterns if re.search(pattern, question_lower))
        
        complexity_score = (high_matches * 3 + medium_matches * 2 - simple_matches * 0.5)
        
        conditional_words = ['if', 'suppose', 'assuming', 'given that', 'provided that']
        conditional_boost = sum(1 for word in conditional_words if word in question_lower) * 0.5
        
        comparative_patterns = [r'\bmore\b', r'\bless\b', r'\bbetter\b', r'\bworse\b', r'\bmost\b', r'\bleast\b']
        comparative_boost = sum(1 for pattern in comparative_patterns if re.search(pattern, question_lower)) * 0.3
        
        total_score = complexity_score + conditional_boost + comparative_boost
        return 1 / (1 + math.exp(-total_score + 1))
    
    def _calculate_domain_complexity(self, question):
        question_lower = question.lower()
        
        agriculture_domains = {
            'crop_science': ['crop', 'cultivation', 'planting', 'harvest', 'yield', 'seeds', 'germination'],
            'soil_science': ['soil', 'fertility', 'nutrients', 'nitrogen', 'phosphorus', 'potassium', 'ph'],
            'pest_management': ['pest', 'disease', 'insect', 'fungus', 'virus', 'bacteria', 'weed'],
            'water_management': ['irrigation', 'water', 'drought', 'rainfall', 'precipitation'],
            'climate_weather': ['climate', 'weather', 'temperature', 'humidity', 'frost'],
            'technology_precision': ['precision agriculture', 'gps', 'remote sensing', 'drones', 'sensors'],
            'economics_policy': ['subsidy', 'market', 'price', 'cost', 'profit', 'economics'],
            'livestock_animal': ['livestock', 'cattle', 'dairy', 'poultry', 'pig', 'sheep'],
            'post_harvest': ['storage', 'processing', 'packaging', 'transportation']
        }
        
        domain_weights = {
            'technology_precision': 3.0, 'soil_science': 2.5, 'pest_management': 2.5,
            'climate_weather': 2.0, 'water_management': 2.0, 'economics_policy': 2.0,
            'crop_science': 1.5, 'livestock_animal': 1.5, 'post_harvest': 1.5
        }
        
        domain_scores = []
        for domain, terms in agriculture_domains.items():
            matches = sum(1 for term in terms if term in question_lower)
            if matches > 0:
                weight = domain_weights.get(domain, 1.0)
                score = math.log(1 + matches) * weight
                domain_scores.append(score)
        
        if not domain_scores:
            return 0.0
            
        max_score = max(domain_scores)
        diversity_penalty = len(domain_scores) / len(agriculture_domains)
        
        technical_terms = ['agroecology', 'biotechnology', 'genomics', 'phenotyping', 'bioinformatics']
        tech_boost = sum(1 for term in technical_terms if term in question_lower) * 0.5
        
        total_score = max_score + diversity_penalty + tech_boost
        return math.tanh(total_score / 3)
    
    def _calculate_semantic_complexity(self, question):
        question_lower = question.lower()
        words = question.split()
        
        technical_indicators = [
            'physiological', 'biochemical', 'molecular', 'cellular', 'genetic',
            'morphological', 'phenotypic', 'genotypic', 'metabolic', 'enzymatic'
        ]
        
        tech_density = sum(1 for term in technical_indicators if term in question_lower) / len(words)
        
        measurement_patterns = [
            r'\d+\s*(kg|ton|hectare|acre|liter|ml|ppm|ph|°c|°f)',
            r'\d+%\s*(moisture|protein|fat|fiber)',
            r'\d+\s*(days|weeks|months)\s*(after|before)'
        ]
        
        measurement_count = sum(1 for pattern in measurement_patterns if re.search(pattern, question_lower))
        measurement_factor = math.log(1 + measurement_count)
        
        proper_nouns = sum(1 for word in words if word[0].isupper() and len(word) > 3)
        entity_density = proper_nouns / len(words)
        
        concept_separators = ['crop and soil', 'pest and disease', 'irrigation and fertilization']
        concept_complexity = sum(1 for sep in concept_separators if sep in question_lower)
        
        feature_vector = np.array([tech_density * 10, measurement_factor, entity_density * 10, concept_complexity])
        weights = np.array([0.4, 0.2, 0.2, 0.2])
        
        semantic_score = np.dot(feature_vector, weights)
        return math.tanh(semantic_score)
    
    def _calculate_intent_complexity(self, question):
        question_lower = question.lower()
        
        intent_patterns = {
            'diagnostic': [r'diagnose\b', r'identify.*problem\b', r'symptoms\b', r'deficiency\b'],
            'optimization': [r'optimize\b', r'maximize.*yield\b', r'improve.*productivity\b'],
            'planning': [r'plan.*cultivation\b', r'schedule.*planting\b', r'timing\b'],
            'comparative': [r'compare.*varieties\b', r'best.*cultivar\b', r'versus\b'],
            'predictive': [r'predict.*yield\b', r'forecast.*weather\b', r'estimate.*production\b'],
            'management': [r'manage\b', r'control\b', r'prevent\b', r'strategy\b']
        }
        
        intent_weights = {
            'diagnostic': 3.0, 'optimization': 2.8, 'predictive': 2.5,
            'planning': 2.0, 'management': 1.8, 'comparative': 1.5
        }
        
        intent_scores = []
        for intent_type, patterns in intent_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, question_lower))
            if matches > 0:
                weight = intent_weights.get(intent_type, 1.0)
                score = math.log(1 + matches) * weight
                intent_scores.append(score)
        
        if not intent_scores:
            return 0.0
            
        max_intent_score = max(intent_scores)
        
        process_words = ['preparation', 'planting', 'maintenance', 'harvesting', 'post-harvest']
        process_complexity = sum(1 for word in process_words if word in question_lower)
        process_factor = math.sqrt(process_complexity) * 0.3
        
        total_score = max_intent_score + process_factor
        return math.tanh(total_score / 3)
    
    def _calculate_document_quality(self, question, documents):
        if not documents:
            return 0.0
            
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_words = question_words - stop_words
        
        quality_scores = []
        
        for doc in documents[:5]:
            doc_content = doc.page_content.lower()
            doc_words = set(re.findall(r'\b\w+\b', doc_content))
            
            if not question_words:
                relevance_ratio = 0
            else:
                overlap = len(question_words.intersection(doc_words))
                relevance_ratio = overlap / len(question_words)
            
            content_length = len(doc.page_content)
            length_factor = math.log(1 + content_length) / math.log(1000)
            
            sentences = doc_content.split('.')
            relevant_sentences = sum(1 for sent in sentences if any(word in sent for word in question_words))
            context_richness = relevant_sentences / max(len(sentences), 1)
            
            doc_quality = (relevance_ratio * 0.5 + length_factor * 0.3 + context_richness * 0.2)
            quality_scores.append(doc_quality)
        
        if not quality_scores:
            return 0.0
            
        avg_quality = np.mean(quality_scores)
        return math.tanh(avg_quality * 2)
    
    def _route_decision(self, complexity_score, doc_quality, question):
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what is', 'define', 'meaning of']):
            threshold_simple, threshold_complex = 0.3, 0.7
        elif any(word in question_lower for word in ['how to', 'best practice', 'recommend']):
            threshold_simple, threshold_complex = 0.2, 0.6
        elif any(word in question_lower for word in ['diagnose', 'problem', 'disease', 'pest']):
            threshold_simple, threshold_complex = 0.15, 0.5
        elif any(word in question_lower for word in ['yield', 'productivity', 'optimize', 'maximize']):
            threshold_simple, threshold_complex = 0.1, 0.45
        else:
            threshold_simple, threshold_complex = 0.25, 0.65
        
        combined_score = complexity_score * (1 - doc_quality * 0.3)
        
        if combined_score <= threshold_simple and doc_quality >= 0.7:
            return "simple_fast"
        elif combined_score >= threshold_complex or doc_quality <= 0.2:
            return "complex"
        else:
            return "moderate"

    # === ADAPTIVE WORKFLOW HANDLERS ===
    
    def adaptive_router(self, state):
        complexity_level = self.analyze_query_complexity(state)
        
        if complexity_level == "simple_fast":
            return "simple_path"
        elif complexity_level == "moderate":
            return "moderate_path"
        else:
            return "complex_path"
    
    def simple_query_handler(self, state):
        """Handle simple queries with fast generation"""
        print("---SIMPLE QUERY DETECTED - FAST PATH---")
        question = state["question"]
        documents = state["documents"]
        
        simple_prompt = f"""
        Based on the provided context, give a direct and concise answer to the question.
        
        Context: {documents[0].page_content if documents else "No context available"}
        Question: {question}
        
        Answer:
        """
        
        generation = self.llm.invoke(simple_prompt).content
        
        if len(self.chat_memory) < 5:
            self.chat_memory.append(generation)
        else:
            self.chat_memory.pop(0)
            self.chat_memory.append(generation)
            
        return {"documents": documents, "question": question, "generation": generation, "workflow_type": "simple"}
        
    def moderate_query_handler(self, state):
        print("---MODERATE QUERY DETECTED - STANDARD RAG---")
        return self.generate(state)
        
    def complex_query_handler(self, state):
        print("---COMPLEX QUERY DETECTED - ENHANCED WORKFLOW---")
        question = state["question"]

        # Step 1: Web search for additional context
        search = Search(self.k)
        try:
            # The Search agent's web_search returns a string summary, not a list of Document objects.
            web_search_result = search.web_search(question)
            # Wrap the result in a Document object for consistency
            from langchain_core.documents import Document
            if web_search_result and isinstance(web_search_result, str):
                web_docs = [Document(page_content=web_search_result, metadata={"source": "web_search"})]
            else:
                web_docs = []
        except Exception as e:
            print(f"Error during web search: {e}")
            web_docs = []

        print(f"Retrieved {len(web_docs)} web documents")

        # Step 2: Combine retrieved docs with web search results
        docs_list = state["documents"] if isinstance(state["documents"], list) else [state["documents"]]
        docs_list = [doc for doc in docs_list if doc and hasattr(doc, "page_content") and doc.page_content]
        combined_docs = docs_list + web_docs[:3]  # Add up to 3 web results

        abstractor = Abstractor(self.model)
        try:
            extractions = abstractor.abstract(combined_docs)
        except Exception as e:
            print(f"Error during abstraction: {e}")
            extractions = ""

        print("extracted info: ", extractions)

        enhanced_prompt = f"""
        You are answering a complex agricultural query that requires comprehensive analysis. 
        Use all provided context including web search results and extracted information.

        Question: {question}
        Retrieved Documents: {docs_list[:2]}
        Web Search Results: {web_docs[:2] if web_docs else "None"}
        Extracted Key Information: {extractions}
        Chat History: {self.chat_memory}

        Provide a detailed, well-structured answer that addresses all aspects of the question.
        """

        try:
            initial_generation = self.llm.invoke(enhanced_prompt).content
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            initial_generation = "Error generating answer."

        hallucinationGrader = HallucinationGrader(self.model)
        try:
            hallucination_grade = hallucinationGrader.grade_hallucinations(combined_docs, initial_generation)
        except Exception as e:
            print(f"Error during hallucination grading: {e}")
            hallucination_grade = "no"

        answerGrader = AnswerGrader(self.model)
        try:
            answer_grade = answerGrader.grade_answer(question, initial_generation)
        except Exception as e:
            print(f"Error during answer grading: {e}")
            answer_grade = "no"

        print(f"Initial generation grading - Hallucination: {hallucination_grade}, Answer quality: {answer_grade}")

        if hallucination_grade == "no" or answer_grade == "no":
            print("---USING INTROSPECTIVE AGENT FOR COMPLEX QUERY---")

            introspective_prompt = f"""
            This is a complex agricultural question that requires careful analysis: {question}

            Previous attempt generated: {initial_generation}

            Issues identified:
            - Hallucination check: {hallucination_grade}
            - Answer quality: {answer_grade}

            Available context:
            - Original documents: {docs_list}
            - Web search results: {web_docs[:2] if web_docs else "None"}
            - Extracted information: {extractions}

            Please provide a more accurate and comprehensive answer, ensuring it's grounded in the provided context.
            """

            try:
                introspective_response = self.introspective_agent.introspect_and_respond(introspective_prompt)
                final_generation = str(introspective_response)
            except Exception as e:
                print(f"Error during introspective agent response: {e}")
                final_generation = initial_generation
        else:
            final_generation = initial_generation

        if len(self.chat_memory) < 5:
            self.chat_memory.append(final_generation)
        else:
            self.chat_memory.pop(0)
            self.chat_memory.append(final_generation)

        return {
            "documents": combined_docs,
            "question": question,
            "generation": final_generation,
            "extractions": extractions,
            "workflow_type": "complex",
            "initial_generation": initial_generation,
            "used_introspective_agent": hallucination_grade == "no" or answer_grade == "no"
        }