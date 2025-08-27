import os
import sys
from typing import Dict, Any, List, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END, START

from RAG.workflow import Workflow
from RAG.parallel_rag_main import ParallelRAGSystem
from Agents.Router import RouterAgent
from Agents.Crop_Recommender.agent import CropRecommenderAgent
from Agents.Weather_forcast.agent import WeatherForecastAgent
from Agents.Location_Information.agent import LocationAgriAssistant
from Agents.News.agent import NewsAgent
from Agents.Credit_Policy_Market.agent import CreditPolicyMarketAgent
from Agents.answer_grader import AnswerGraderAgent
from Agents.synthesizer_agent import SynthesizerAgent
from Agents.Crop_Disease.agent import CropDiseaseAgent
from Agents.fact_checker.fscorer import LikertScorer
from Agents.Image_Analysis.agent import ImageAgent
from Agents.Market_Price.agent import MarketPriceAgent
from Agents.Multi_Lingual.agent import MultiLingualAgent
from Agents.Pest_prediction.agent import PestPredictionAgent
from Agents.Risk_Management.agent import AgriculturalRiskAnalysisAgent
from Agents.Web_Scrapping.agent import AgriculturalWebScrappingAgent
from Agents.Crop_Yield.agent import CropYieldAssistant
from Agents.Query_rewriter import QueryRewriterAgent
from Agents.Chart_Agent.agent import AgriculturalChartAgent
from Agents.Fertilizer_Recommender.agent import FertilizerRecommendationAgent
from Agents.Guardrails.agent import AgriculturalGuardrailsAgent
from utils.Internet_checker import InternetChecker
from utils.hf_model import HFModel

internet_checker = InternetChecker()
base_model_dir = "./models/Qwen1.5-Base"
adapter_dir = "./models/Qwen_1.5_Finetuned"
hf_model = None

if not internet_checker.is_connected():
    print("Offline mode detected. Using HF Model for inference.")
    hf_model = HFModel(base_model_dir, adapter_dir)
else: 
    print("Online mode detected")

guardrails_agent = AgriculturalGuardrailsAgent()
crop_recommender_agent = CropRecommenderAgent()
weather_forecast_agent = WeatherForecastAgent()
location_agri_assistant = LocationAgriAssistant()
news_agent = NewsAgent()
chart_agent = AgriculturalChartAgent()
crop_yield_assistant = CropYieldAssistant()
credit_policy_market_agent = CreditPolicyMarketAgent()
answer_grader_agent = AnswerGraderAgent()
synthesizer_agent = SynthesizerAgent()
crop_disease_agent = CropDiseaseAgent()
fact_checker_agent = LikertScorer()
image_analysis_agent = ImageAgent()
market_price_agent = MarketPriceAgent()
multi_language_translator_agent = MultiLingualAgent()
pest_prediction_agent = PestPredictionAgent()
risk_management_agent = AgriculturalRiskAnalysisAgent()
web_scraping_agent = AgriculturalWebScrappingAgent()
query_rewriter_agent = QueryRewriterAgent()
fertilizer_recommender_agent = FertilizerRecommendationAgent()

class MainWorkflowState(TypedDict):
    query: str
    image_path: str
    initial_mode: str
    current_mode: str
    rag_response: str
    router_result: Dict[str, Any]
    agent_responses: Dict[str, Any]
    fact_checked_responses: Dict[str, Any]
    synthesized_result: str
    grading: Any
    answer_grade: Any
    rewritten_query: str
    generation: str
    extractions: str
    documents: List[str]
    has_switched_mode: bool
    is_image_query: bool
    chart_path: str
    chart_extra_message: str
    guardrails_result: Dict[str, Any]
    is_agriculture_related: bool
    guardrails_response: str

def run_adaptive_rag(query: str) -> str:
    rag_system = ParallelRAGSystem(model="gemini-2.0-flash", k=3)
    result = rag_system.process_query(query)
    return result.get("synthesized_answer", "")

def run_router_agent(query: str, image_path: str = None) -> Dict[str, Any]:
    router = RouterAgent()
    if image_path:
        query_with_image = f"{query} [IMAGE_PROVIDED]"
        routing_decision = router.route(query_with_image)
    else:
        routing_decision = router.route(query)
    
    if hasattr(routing_decision, 'agents'):
        agents = routing_decision.agents
    elif isinstance(routing_decision, dict):
        agents = routing_decision.get('agents', [])
    else:
        agents = []
    return {"agents": agents, "routing_decision": routing_decision}

def call_agent(agent_name: str, query: str, image_path: str = None) -> Any:
    if agent_name == "CropRecommenderAgent":
        return crop_recommender_agent.respond(query)
    elif agent_name == "WeatherForecastAgent":
        return weather_forecast_agent.get_weather_analysis(query)
    elif agent_name == "LocationAgriAssistant":
        return location_agri_assistant.respond(query)
    elif agent_name == "NewsAgent":
        return news_agent.get_agri_news(query)
    elif agent_name == "CreditPolicyMarketAgent":
        return credit_policy_market_agent.respond_to_query(query)
    elif agent_name == "AnswerGraderAgent":
        return answer_grader_agent.grade(query.get("question", ""), query.get("answer", ""))
    elif agent_name == "SynthesizerAgent":
        return synthesizer_agent.synthesize(query.get("responses", []))
    elif agent_name == "CropDiseaseDetectionAgent":
        return crop_disease_agent.analyze_disease(query = query, image_path=image_path)
    elif agent_name == "FactCheckerAgent":
        return fact_checker_agent.score(query)
    elif agent_name == "ImageAnalysisAgent":
        return image_analysis_agent.describe_image(image_path)
    elif agent_name == "MarketPriceAgent":
        return market_price_agent.chat(query)
    elif agent_name == "MultiLanguageTranslatorAgent":
        return multi_language_translator_agent.translate_robust(query)
    elif agent_name == "PestPredictionAgent":
        return pest_prediction_agent.respond(query)
    elif agent_name == "RiskManagementAgent":
        return risk_management_agent.assess(query)
    elif agent_name == "WebScrapingAgent":
        return web_scraping_agent.scrape(query)
    elif agent_name == "CropYieldAgent":
        return crop_yield_assistant.respond(query)
    elif agent_name == "FertilizerRecommenderAgent":
        return fertilizer_recommender_agent.recommend_fertilizer(query)
    elif agent_name == "ChartAgent":
        return chart_agent.generate_response(query)
    else:
        return f"No implementation for agent: {agent_name}"

def call_agent_simple(agent_name: str, query: str, image_path: str = None) -> Dict[str, Any]:
    try:
        agent_response = call_agent(agent_name, query, image_path)
        
        if agent_name == "ChartAgent" and agent_response:
            chart_path = ""
            extra_message = ""

            if hasattr(agent_response, 'imagekit_url'):
                chart_path = agent_response.imagekit_url
            elif isinstance(agent_response, dict) and 'imagekit_url' in agent_response:
                chart_path = agent_response['imagekit_url']

            if hasattr(agent_response, 'extra_message'):
                extra_message = agent_response.extra_message
            elif isinstance(agent_response, dict) and 'extra_message' in agent_response:
                extra_message = agent_response['extra_message']
            
            response_text = f"{extra_message}\n\nChart generated at: {chart_path}" if chart_path else extra_message
            
            return {
                "agent_name": agent_name,
                "response": response_text,
                "chart_path": chart_path,
                "extra_message": extra_message
            }
        
        return {
            "agent_name": agent_name,
            "response": agent_response,
        }
    except Exception as e:
        return {
            "agent_name": agent_name,
            "response": f"Error: {str(e)}",
        }

def grade_answer(question: str, answer: str) -> Dict[str, Any]:
    try:
        grade_result = answer_grader_agent.grade(question, answer)
        print(f"Answer Grader Response: {grade_result}")
        return {
            "grade": grade_result.decision,
            "is_good_answer": getattr(grade_result, 'decision', False),
            "reasoning": getattr(grade_result, 'feedback', ''),
            "score": 1 if getattr(grade_result, 'decision', False) else 0
        }
    except Exception as e:
        return {
            "grade": None,
            "is_good_answer": False,
            "reasoning": f"Grading error: {str(e)}",
            "score": 0,
            "error": str(e)
        }

def guardrails_node(state: MainWorkflowState):
    try:
        guardrails_result = guardrails_agent.evaluate_query(state["query"])
        print(f"Guardrails evaluation: {guardrails_result}")
        
        return {
            "guardrails_result": {
                "is_agriculture_related": guardrails_result.is_agriculture_related,
                "is_greeting": guardrails_result.is_greeting,
                "response_message": guardrails_result.response_message,
                "confidence_score": guardrails_result.confidence_score,
                "category": guardrails_result.category
            },
            "is_agriculture_related": guardrails_result.is_agriculture_related,
            "guardrails_response": guardrails_result.response_message or ""
        }
    except Exception as e:
        print(f"Error in guardrails_node: {str(e)}")
        return {
            "guardrails_result": {
                "is_agriculture_related": False,
                "is_greeting": True,
                "response_message": "Hello! I'm here to help you with all your agricultural needs.",
                "confidence_score": 0.5,
                "category": "greeting"
            },
            "is_agriculture_related": False,
            "guardrails_response": "Hello! I'm here to help you with all your agricultural needs."
        }

def rag_node(state: MainWorkflowState):
    rag_response = run_adaptive_rag(state["query"])
    documents = []
    extractions = ""
    if isinstance(rag_response, dict):
        documents = rag_response.get("documents", [])
        extractions = rag_response.get("extractions", "")
        generation = rag_response.get("generation", "")
    else:
        generation = rag_response
    
    return {
        "rag_response": rag_response,
        "synthesized_result": rag_response,
        "documents": documents,
        "extractions": extractions,
        "generation": generation,
        "current_mode": "rag"
    }

def router_node(state: MainWorkflowState):
    router_result = run_router_agent(state["query"], state.get("image_path"))
    return {
        "router_result": router_result,
        "current_mode": "tooling"
    }

def agent_calls_node(state: MainWorkflowState):
    agent_responses = {}
    chart_path = ""
    chart_extra_message = ""
    agents = state["router_result"].get("agents", [])
    
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(call_agent_simple, agent, state["query"], state.get("image_path")): agent 
            for agent in agents
        }
        
        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result()
                agent_responses[agent_name] = result["response"]
                print(f"Agent {agent_name} Response: {result['response']}")
                
                if agent_name == "ChartAgent":
                    print("Agent ChartAgent Response:", result)
                    agent_responses[agent_name] = result["extra_message"]
                    chart_path = result.get("chart_path", "")
                    chart_extra_message = result.get("extra_message", "")
                
            except Exception as e:
                agent_responses[agent_name] = f"Error: {str(e)}"
                print(f"Agent {agent_name} Error: {str(e)}")
    
    return {
        "agent_responses": agent_responses,
        "chart_path": chart_path,
        "chart_extra_message": chart_extra_message
    }

def synthesize_tooling_node(state: MainWorkflowState):
    all_responses = []
    
    for agent_name, response in state["agent_responses"].items():
        all_responses.append(response)
    
    synthesized_result = synthesizer_agent.synthesize(all_responses)
    
    if state.get("chart_path") and state.get("chart_extra_message"):
        synthesized_result = f"{state['chart_extra_message']}\n\n{synthesized_result}\n\nChart available at: {state['chart_path']}"
    
    return {
        "synthesized_result": synthesized_result
    }

def grading_node(state: MainWorkflowState):
    answer_grade_result = grade_answer(state["query"], str(state["synthesized_result"]))
    
    return {
        "answer_grade": answer_grade_result
    }

def guardrails_routing_edge(state: MainWorkflowState):
    is_agriculture_related = state.get("is_agriculture_related", False)
    is_greeting = state["guardrails_result"].get("is_greeting", False)
    
    print(f"Guardrails routing: agriculture={is_agriculture_related}, greeting={is_greeting}")
    
    if not is_agriculture_related and not is_greeting:
        print("Routing to: guardrails_end")
        return "guardrails_end"
    
    if is_greeting:
        print("Routing to: greeting_end")
        return "greeting_end"
    
    if state.get("is_image_query", False):
        print("Routing to: router (image query)")
        return "router"
    
    initial_mode = state["initial_mode"]
    if initial_mode == "rag":
        print("Routing to: rag")
        return "rag"
    else:
        print("Routing to: router")
        return "router"

def mode_decision_edge(state: MainWorkflowState):
    if state.get("is_image_query", False):
        return "end"
    
    answer_grade = state["answer_grade"]
    current_mode = state["current_mode"]
    has_switched_mode = state.get("has_switched_mode", False)
    
    is_answer_complete = answer_grade.get("is_good_answer", False)
    
    print(f"Quality Assessment - Answer Complete: {is_answer_complete}")
    
    if is_answer_complete:
        return "end"
    
    if current_mode == "tooling":
        return "fallback"
    elif current_mode == "rag":
        return "switch_to_tooling"
    
    return "fallback"

def switch_to_tooling_node(state: MainWorkflowState):
    router_result = run_router_agent(state["query"], state.get("image_path"))
    return {
        "router_result": router_result,
        "current_mode": "tooling",
        "has_switched_mode": True
    }

def fallback_node(state: MainWorkflowState):
    fallback_response = multi_language_translator_agent.respond(f"Provide a general answer for: {state['query']}")
    
    return {
        "synthesized_result": fallback_response
    }

def guardrails_response_node(state: MainWorkflowState):
    guardrails_response = state.get("guardrails_response", "")
    if not guardrails_response:
        guardrails_response = "I specialize in agricultural assistance. I can help you with farming practices, crop management, weather forecasting, market prices, and other agriculture-related topics. Is there anything farming-related I can help you with?"
    
    return {
        "synthesized_result": guardrails_response
    }

def greeting_response_node(state: MainWorkflowState):
    greeting_response = state.get("guardrails_response", "")
    if not greeting_response:
        greeting_response = "Hello! I'm here to help you with all your agricultural needs. Whether you have questions about crop cultivation, pest management, weather forecasting, or market prices, I'm ready to assist. How can I help you with your farming today?"
    
    return {
        "synthesized_result": greeting_response
    }

def build_hybrid_workflow_graph():
    graph = StateGraph(MainWorkflowState)
    
    graph.add_node("guardrails", guardrails_node)
    graph.add_node("guardrails_response", guardrails_response_node)
    graph.add_node("greeting_response", greeting_response_node)
    graph.add_node("rag", rag_node)
    graph.add_node("router", router_node)
    graph.add_node("agent_calls", agent_calls_node)
    graph.add_node("synthesize_tooling", synthesize_tooling_node)
    graph.add_node("grading", grading_node)
    graph.add_node("switch_to_tooling", switch_to_tooling_node)
    graph.add_node("fallback", fallback_node)
    
    graph.add_edge(START, "guardrails")
    
    graph.add_conditional_edges(
        "guardrails", 
        guardrails_routing_edge, 
        {
            "guardrails_end": "guardrails_response",
            "greeting_end": "greeting_response",
            "rag": "rag", 
            "router": "router"
        }
    )
    
    graph.add_edge("guardrails_response", END)
    graph.add_edge("greeting_response", END)
    
    graph.add_edge("rag", "grading")
    graph.add_edge("router", "agent_calls")
    graph.add_edge("agent_calls", "synthesize_tooling")
    graph.add_edge("synthesize_tooling", "grading")
    
    graph.add_conditional_edges(
        "grading", 
        mode_decision_edge, 
        {
            "end": END, 
            "switch_to_tooling": "switch_to_tooling",
            "fallback": "fallback"
        }
    )
    
    graph.add_edge("switch_to_tooling", "agent_calls")
    graph.add_edge("fallback", END)
    
    return graph

hybrid_workflow_graph = build_hybrid_workflow_graph()
compiled_hybrid_graph = hybrid_workflow_graph.compile()

def run_workflow(query: str, mode: str = "rag", image_path: str = None) -> Dict[str, Any]:
    if not internet_checker.is_connected() and hf_model:
        hf_response = hf_model.infer(query)
        return {
            "answer": hf_response,
            "answer_quality_grade": {"is_good_answer": True, "reasoning": "Offline mode - HF Model response"},
            "is_answer_complete": True,
            "final_mode": "offline",
            "switched_modes": False,
            "is_image_query": image_path is not None,
            "chart_path": "",
            "chart_extra_message": "",
            "is_agriculture_related": True,
            "guardrails_passed": True,
            "guardrails_category": "agriculture",
            "guardrails_confidence": 1.0
        }
    
    is_image_query = image_path is not None
    
    state = MainWorkflowState(
        query=query,
        image_path=image_path or "",
        initial_mode=mode,
        current_mode=mode,
        rag_response="",
        router_result={},
        agent_responses={},
        fact_checked_responses={},
        synthesized_result="",
        grading=None,
        answer_grade=None,
        rewritten_query="",
        generation="",
        extractions="",
        documents=[],
        has_switched_mode=False,
        is_image_query=is_image_query,
        chart_path="",
        chart_extra_message="",
        guardrails_result={},
        is_agriculture_related=False,
        guardrails_response=""
    )
    
    if mode.lower() not in ["rag", "tooling"]:
        raise ValueError("Mode must be either 'rag' or 'tooling'")
    
    try:
        final_state = compiled_hybrid_graph.invoke(state)
        
        if final_state is None:
            return {
                "answer": "Error: Workflow execution failed",
                "answer_quality_grade": {"is_good_answer": False, "reasoning": "Workflow execution error"},
                "is_answer_complete": False,
                "final_mode": mode,
                "switched_modes": False,
                "is_image_query": is_image_query,
                "chart_path": "",
                "chart_extra_message": "",
                "is_agriculture_related": False,
                "guardrails_passed": False,
                "guardrails_category": "error",
                "guardrails_confidence": 0.0
            }
        
        answer_grade = final_state.get("answer_grade") or {}
        guardrails_result = final_state.get("guardrails_result") or {}
        
        return {
            "answer": final_state.get("synthesized_result", "No response generated"),
            "answer_quality_grade": answer_grade,
            "is_answer_complete": answer_grade.get("is_good_answer", True),
            "final_mode": final_state.get("current_mode", mode),
            "switched_modes": final_state.get("has_switched_mode", False),
            "is_image_query": final_state.get("is_image_query", False),
            "chart_path": final_state.get("chart_path", ""),
            "chart_extra_message": final_state.get("chart_extra_message", ""),
            "is_agriculture_related": final_state.get("is_agriculture_related", False),
            "guardrails_passed": final_state.get("is_agriculture_related", False) or guardrails_result.get("is_greeting", False),
            "guardrails_category": guardrails_result.get("category", ""),
            "guardrails_confidence": guardrails_result.get("confidence_score", 0.0)
        }
        
    except Exception as e:
        print(f"Workflow execution error: {str(e)}")
        return {
            "answer": f"Error executing workflow: {str(e)}",
            "answer_quality_grade": {"is_good_answer": False, "reasoning": f"Workflow error: {str(e)}"},
            "is_answer_complete": False,
            "final_mode": mode,
            "switched_modes": False,
            "is_image_query": is_image_query,
            "chart_path": "",
            "chart_extra_message": "",
            "is_agriculture_related": False,
            "guardrails_passed": False,
            "guardrails_category": "error",
            "guardrails_confidence": 0.0
        }

if __name__ == "__main__":
    import time
    questions = [
        "Hello how are you?",
        "Estimate crop yield for wheat in Punjab in winter of 2025",
        "How can farmers manage pest outbreaks in cotton fields?",
        "What is the market price trend for wheat in India?",
        "How to prevent fungal diseases in tomato crops?",
        "Create a chart showing corn price trends over the last year",
        "Generate a graph for wheat yield predictions",
        "Tell me about cryptocurrency",
        "What's the capital of France?",
        "How to cook pasta?"
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
        print(f"  - Guardrails Passed: {result['guardrails_passed']}")
        print(f"  - Agriculture Related: {result['is_agriculture_related']}")
        print(f"  - Guardrails Category: {result['guardrails_category']}")
        print(f"  - Guardrails Confidence: {result['guardrails_confidence']:.2f}")
        
        if result.get("chart_path"):
            print(f"  - Chart Generated: {result['chart_path']}")
        
        if result.get("chart_extra_message"):
            print(f"  - Chart Insights: {result['chart_extra_message']}")
        
        if result['answer_quality_grade'].get('reasoning'):
            print(f"  - Quality Grade Reasoning: {result['answer_quality_grade']['reasoning']}")