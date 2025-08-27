import sys
import os
import requests
import json
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse
from dataclasses import dataclass
import re
from concurrent.futures import ThreadPoolExecutor
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.tools.tavily import TavilyTools
from Tools.web_scrapper import scrape_agri_prices, scrape_policy_updates, scrape_links

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class Citation:
    title: str
    authors: str
    journal: str
    year: str
    url: str
    source_type: str
    relevance: int
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = None

    def to_apa(self) -> str:
        return f"{self.authors} ({self.year}). {self.title}. {self.journal}."

class EnhancedCitationAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[TavilyTools(), scrape_agri_prices, scrape_policy_updates, scrape_links],
            instructions=self._get_enhanced_instructions(),
            markdown=True,
        )
        
        self.trusted_domains = [
            'nature.com', 'sciencedirect.com', 'springer.com', 'wiley.com',
            'tandfonline.com', 'cambridge.org', 'oxford.com', 'plos.org',
            'usda.gov', 'fao.org', 'who.int', 'cgiar.org', 'worldbank.org',
            'nih.gov', 'pubmed.ncbi.nlm.nih.gov', 'researchgate.net',
            'frontiersin.org', 'mdpi.com', 'elsevier.com', 'jstor.org',
            'journals.ashs.org', 'link.springer.com', 'journals.plos.org',
            'onlinelibrary.wiley.com', 'academic.oup.com', 'emerald.com'
        ]
        
        self.source_quality_scores = {
            'journal_article': 10,
            'conference_paper': 8,
            'government_report': 9,
            'university_publication': 8,
            'extension_publication': 7,
            'industry_report': 6,
            'news_article': 4,
            'blog_post': 3
        }

    def _get_enhanced_instructions(self) -> str:
        return """
        You are an advanced agricultural research citation specialist with enhanced capabilities.

        SEARCH STRATEGY:
        1. Use TavilyTools for comprehensive academic and research searches
        2. Use scrape_links to extract citations from institutional research pages
        3. Use scrape_policy_updates for government agricultural research and policies
        4. Use scrape_agri_prices for agricultural economic research and market studies

        SEARCH EXECUTION:
        - Execute multiple targeted searches using different keywords and phrases
        - Search academic journals, government databases, and institutional repositories
        - Focus on peer-reviewed sources with DOIs when possible
        - Prioritize recent publications (last 5 years) unless specified otherwise
        - Cross-reference findings across multiple sources

        CITATION EXTRACTION:
        - Extract complete bibliographic information
        - Identify and validate DOIs and URLs
        - Capture abstracts when available
        - Determine source quality and type
        - Rate relevance to the research topic (1-10 scale)

        OUTPUT FORMAT:
        For each citation found, provide:
        - Title: [Complete title]
        - Authors: [Full author list]
        - Journal: [Publication venue]
        - Year: [Publication year]
        - URL: [Direct link or DOI]
        - Type: [Source type]
        - Relevance: [1-10 score]
        - Abstract: [If available]

        QUALITY STANDARDS:
        - Only include citations from reputable sources
        - Verify all URLs are accessible
        - Ensure direct relevance to agricultural research
        - Prioritize peer-reviewed and authoritative sources
        """

    def _validate_url_batch(self, urls: List[str]) -> Dict[str, bool]:
        def check_url(url: str) -> Tuple[str, bool]:
            try:
                if not url or not url.startswith(('http://', 'https://')):
                    return url, False
                
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower().replace('www.', '')
                
                is_trusted = any(trusted in domain for trusted in self.trusted_domains)
                if not is_trusted:
                    return url, False
                
                response = requests.head(url, timeout=10, allow_redirects=True)
                return url, response.status_code == 200
                
            except Exception:
                return url, False

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(check_url, urls))
        
        return dict(results)

    def _extract_citations_enhanced(self, response_text: str) -> List[Citation]:
        citations = []
        
        citation_patterns = [
            r'Title:\s*(.+?)(?:\n|$)',
            r'\*\*Title\*\*:\s*(.+?)(?:\n|$)',
            r'# (.+?)(?:\n|$)'
        ]
        
        author_patterns = [
            r'Authors?:\s*(.+?)(?:\n|$)',
            r'\*\*Authors?\*\*:\s*(.+?)(?:\n|$)'
        ]
        
        journal_patterns = [
            r'Journal:\s*(.+?)(?:\n|$)',
            r'Source:\s*(.+?)(?:\n|$)',
            r'\*\*Journal\*\*:\s*(.+?)(?:\n|$)'
        ]
        
        year_patterns = [
            r'Year:\s*(\d{4})',
            r'\*\*Year\*\*:\s*(\d{4})',
            r'\((\d{4})\)'
        ]
        
        url_patterns = [
            r'URL:\s*(https?://[^\s\n]+)',
            r'DOI:\s*(https?://[^\s\n]+)',
            r'Link:\s*(https?://[^\s\n]+)',
            r'(https?://[^\s\n]+)'
        ]
        
        sections = re.split(r'\n\s*\n', response_text)
        
        for section in sections:
            if len(section.strip()) < 50:
                continue
                
            citation_data = {}
            
            for pattern in citation_patterns:
                match = re.search(pattern, section)
                if match:
                    citation_data['title'] = match.group(1).strip()
                    break
            
            for pattern in author_patterns:
                match = re.search(pattern, section)
                if match:
                    citation_data['authors'] = match.group(1).strip()
                    break
            
            for pattern in journal_patterns:
                match = re.search(pattern, section)
                if match:
                    citation_data['journal'] = match.group(1).strip()
                    break
            
            for pattern in year_patterns:
                match = re.search(pattern, section)
                if match:
                    citation_data['year'] = match.group(1).strip()
                    break
            
            urls_found = []
            for pattern in url_patterns:
                matches = re.findall(pattern, section)
                urls_found.extend(matches)
            
            if urls_found:
                citation_data['url'] = urls_found[0]
            
            relevance_match = re.search(r'Relevance:\s*(\d+)', section)
            relevance = int(relevance_match.group(1)) if relevance_match else 7
            
            type_match = re.search(r'Type:\s*(.+?)(?:\n|$)', section)
            source_type = type_match.group(1).strip() if type_match else 'journal_article'
            
            abstract_match = re.search(r'Abstract:\s*(.+?)(?:\n\n|$)', section, re.DOTALL)
            abstract = abstract_match.group(1).strip() if abstract_match else None
            
            if 'title' in citation_data and len(citation_data) >= 3:
                citation = Citation(
                    title=citation_data.get('title', ''),
                    authors=citation_data.get('authors', 'Unknown'),
                    journal=citation_data.get('journal', 'Unknown'),
                    year=citation_data.get('year', 'Unknown'),
                    url=citation_data.get('url', 'Unknown'),
                    abstract=abstract,
                    relevance=relevance,
                    source_type=source_type
                )
                citations.append(citation)
        
        return citations

    def find_citations_basic(self, topic: str, num_citations: int = 15, 
                            focus_areas: List[str] = None) -> Dict[str, Any]:
        """Basic citation finding method"""
        logger.info(f"Starting basic citation search for: {topic}")
        
        try:
            search_query = f"""
            Find academic citations and research papers for: {topic}
            
            Search requirements:
            - Find {num_citations} high-quality academic citations
            - Focus on peer-reviewed journal articles
            - Include government reports and institutional publications
            - Prioritize recent publications (last 5 years)
            - Extract complete bibliographic information
            """
            
            if focus_areas:
                search_query += f"\nSpecial focus on: {', '.join(focus_areas)}"
            
            response = self.agent.run(search_query)
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            citations = self._extract_citations_enhanced(response_text)
            
            valid_citations = []
            urls_to_check = []
            
            for citation in citations:
                if citation.url and citation.url != 'Unknown':
                    urls_to_check.append(citation.url)
            
            url_validity = {}
            if urls_to_check:
                url_validity = self._validate_url_batch(urls_to_check)
            
            for citation in citations:
                if citation.url == 'Unknown' or url_validity.get(citation.url, True):
                    valid_citations.append(citation)
            
            valid_citations.sort(key=lambda c: c.relevance, reverse=True)
            valid_citations = valid_citations[:num_citations]
            
            logger.info(f"Found {len(valid_citations)} valid citations")
            
            return {
                "success": len(valid_citations) > 0,
                "citations": valid_citations,
                "valid_count": len(valid_citations),
                "total_found": len(citations),
                "search_query": search_query,
                "focus_areas": focus_areas
            }
        except Exception as e:
            logger.error(f"Error during basic citation search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "citations": [],
                "valid_count": 0,
                "total_found": 0
            }

    def find_citations_enhanced(self, topic: str, num_citations: int = 15, 
                               focus_areas: List[str] = None) -> Dict[str, Any]:
        """Enhanced citation finding with comprehensive search across multiple sources"""
        logger.info(f"Starting enhanced citation search for: {topic}")
        
        try:
            search_query = f"""
            Find academic citations and research papers for: {topic}
            
            Search requirements:
            - Find {num_citations} high-quality academic citations
            - Focus on peer-reviewed journal articles
            - Include government reports and institutional publications
            - Prioritize recent publications (last 5 years)
            - Extract complete bibliographic information
            """
            
            if focus_areas:
                search_query += f"\nSpecial focus on: {', '.join(focus_areas)}"
            
            response = self.agent.run(search_query)
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            citations = self._extract_citations_enhanced(response_text)
            
            valid_citations = []
            urls_to_check = []
            
            for citation in citations:
                if citation.url and citation.url != 'Unknown':
                    urls_to_check.append(citation.url)
            
            url_validity = {}
            if urls_to_check:
                url_validity = self._validate_url_batch(urls_to_check)
            
            for citation in citations:
                if citation.url == 'Unknown' or url_validity.get(citation.url, True):
                    valid_citations.append(citation)
            
            valid_citations.sort(key=lambda c: c.relevance, reverse=True)
            valid_citations = valid_citations[:num_citations]
            
            logger.info(f"Found {len(valid_citations)} valid citations")
            
            return {
                "success": len(valid_citations) > 0,
                "citations": valid_citations,
                "valid_count": len(valid_citations),
                "total_found": len(citations),
                "search_query": search_query,
                "focus_areas": focus_areas
            }
        except Exception as e:
            logger.error(f"Error during enhanced citation search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "citations": [],
                "valid_count": 0,
                "total_found": 0
            }
        
if __name__ == "__main__":
    agent = EnhancedCitationAgent()

    result = agent.find_citations_basic(
        topic="Soil Health Improvement in Wheat Farming",
        num_citations=5,
        focus_areas=["sustainability", "fertilizer management"]
    )

    if result["success"]:
        print(f"\n✅ Found {result['valid_count']} valid citations (out of {result['total_found']})\n")
        for i, citation in enumerate(result["citations"], 1):
            print(f"{i}. {citation.to_apa()}")
            print(f"   URL: {citation.url}")
            print(f"   Relevance: {citation.relevance}/10\n")
    else:
        print("\n❌ No citations found.")
        if "error" in result:
            print(f"Error: {result['error']}")
