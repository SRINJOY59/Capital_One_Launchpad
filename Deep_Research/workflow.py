import os
import sys
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging
from datetime import datetime
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import textwrap

from planner_agent import AgriculturalPlanningAgent
from SubsearchAgent import EnhancedSubsearchAgent
from citations_agent import EnhancedCitationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchState:
    title: str = ""
    objective: str = ""
    location: str = "Global"
    focus_areas: List[str] = None
    
    plan: Any = None
    tasks: List[Any] = None
    
    research_results: Dict[str, Any] = None
    citation_results: Dict[str, Any] = None
    
    current_phase: str = "initialized"
    errors: List[str] = None
    execution_log: List[str] = None
    
    def __post_init__(self):
        if self.focus_areas is None:
            self.focus_areas = []
        if self.research_results is None:
            self.research_results = {}
        if self.citation_results is None:
            self.citation_results = {}
        if self.errors is None:
            self.errors = []
        if self.execution_log is None:
            self.execution_log = []

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        custom_styles = {
            'CustomTitle': ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Title'],
                fontSize=18,
                textColor=colors.darkgreen,
                alignment=TA_CENTER,
                spaceAfter=20
            ),
            'SectionHeader': ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.darkblue,
                spaceAfter=12,
                spaceBefore=16
            ),
            'SubHeader': ParagraphStyle(
                name='SubHeader',
                parent=self.styles['Heading3'],
                fontSize=12,
                textColor=colors.darkred,
                spaceAfter=8,
                spaceBefore=12
            ),
            'CustomBodyText': ParagraphStyle(
                name='CustomBodyText',
                parent=self.styles['Normal'],
                fontSize=10,
                alignment=TA_JUSTIFY,
                spaceAfter=6
            ),
            'Citation': ParagraphStyle(
                name='Citation',
                parent=self.styles['Normal'],
                fontSize=9,
                leftIndent=20,
                spaceAfter=4
            ),
            'ResearchContent': ParagraphStyle(
                name='ResearchContent',
                parent=self.styles['Normal'],
                fontSize=10,
                alignment=TA_JUSTIFY,
                spaceAfter=8,
                leftIndent=10
            )
        }
        
        for name, style in custom_styles.items():
            if name not in self.styles:
                self.styles.add(style)
    
    def _sanitize_text(self, text: str) -> str:
        if not text:
            return ""
        text = str(text)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        return text
    
    def generate_pdf_report(self, results: Dict[str, Any], filename: str):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            doc = SimpleDocTemplate(filename, pagesize=A4, 
                                   rightMargin=72, leftMargin=72,
                                   topMargin=72, bottomMargin=18)
            
            story = []
            
            story.extend(self._create_title_page(results))
            story.append(PageBreak())
            
            story.extend(self._create_executive_summary(results))
            story.append(PageBreak())
            
            story.extend(self._create_research_content(results))
            story.append(PageBreak())
            
            story.extend(self._create_citations_section(results))
            
            story.extend(self._create_appendix(results))
            
            doc.build(story)
            logger.info(f"PDF report generated: {filename}")
            
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            raise e
    
    def _create_title_page(self, results: Dict[str, Any]) -> List:
        elements = []
        
        title = self._sanitize_text(results.get('title', 'Agricultural Research Report'))
        elements.append(Paragraph(title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 20))
        
        objective = self._sanitize_text(results.get('objective', ''))
        if objective:
            elements.append(Paragraph(f"<b>Research Objective:</b><br/>{objective}", 
                                    self.styles['Normal']))
            elements.append(Spacer(1, 15))
        
        metadata = [
            ['Location:', self._sanitize_text(results.get('location', 'Global'))],
            ['Generated:', datetime.now().strftime('%B %d, %Y at %H:%M')],
            ['Research Type:', 'Deep Agricultural Analysis'],
        ]
        
        if results.get('focus_areas'):
            focus_text = ', '.join(results['focus_areas'])
            metadata.append(['Focus Areas:', self._sanitize_text(focus_text)])
        
        table = Table(metadata, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 30))
        
        quality = results.get('quality_validation', {})
        if quality:
            quality_text = f"""
            <b>Research Quality Assessment</b><br/>
            Overall Quality: {quality.get('overall_quality', 'Unknown').upper()}<br/>
            Quality Score: {quality.get('quality_score', 0)}/100<br/>
            Citations Found: {quality.get('citations_found', 0)}<br/>
            Sources Analyzed: {quality.get('sources_found', 0)}
            """
            elements.append(Paragraph(quality_text, self.styles['Normal']))
        
        return elements
    
    def _create_executive_summary(self, results: Dict[str, Any]) -> List:
        elements = []
        
        elements.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        
        exec_summary = results.get('execution_summary', {})
        phases_completed = exec_summary.get('phases_completed', 0)
        errors_count = exec_summary.get('errors_encountered', 0)
        
        summary_text = f"""
        This comprehensive agricultural research study was conducted using an automated 
        deep research workflow. The research successfully completed {phases_completed} out of 4 
        planned phases with {errors_count} errors encountered during execution.
        """
        
        elements.append(Paragraph(self._sanitize_text(summary_text.strip()), 
                                self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Key Research Components", self.styles['SubHeader']))
        
        components = []
        
        planning = results.get('planning_results', {})
        if planning.get('plan_generated'):
            components.append(f"Research plan with {planning.get('tasks_created', 0)} tasks generated")
            components.append(f"{planning.get('agent_assignments', 0)} specialist agents assigned")
        
        research = results.get('research_results', {})
        if research and research.get('success'):
            components.append(f"{research.get('successful_searches', 0)} successful research queries")
            components.append(f"{research.get('total_sources_found', 0)} sources analyzed")
        
        citations = results.get('citation_results', {})
        if citations and citations.get('success'):
            components.append(f"{citations.get('valid_count', 0)} validated citations collected")
        
        components.append("Comprehensive research report generated")
        
        for component in components:
            elements.append(Paragraph(self._sanitize_text(component), self.styles['CustomBodyText']))
        
        return elements
    
    def _create_research_content(self, results: Dict[str, Any]) -> List:
        elements = []
        
        elements.append(Paragraph("RESEARCH FINDINGS", self.styles['SectionHeader']))
        
        research_results = results.get('research_results', {})
        if research_results and research_results.get('success'):
            combined_content = research_results.get('combined_content', '')
            search_results = research_results.get('search_results', [])
            
            if combined_content:
                elements.append(Paragraph("Research Summary", self.styles['SubHeader']))
                
                paragraphs = combined_content.split('\n\n')
                for paragraph in paragraphs[:10]:
                    if paragraph.strip():
                        elements.append(Paragraph(self._sanitize_text(paragraph.strip()), 
                                                self.styles['ResearchContent']))
                        elements.append(Spacer(1, 6))
            
            if search_results:
                elements.append(Spacer(1, 12))
                elements.append(Paragraph("Detailed Research Results", self.styles['SubHeader']))
                
                for i, result in enumerate(search_results[:5], 1):
                    if isinstance(result, dict):
                        query = result.get('query', f'Research Query {i}')
                        content = result.get('combined_content', result.get('content', ''))
                        sources_count = result.get('sources_found', 0)
                        
                        elements.append(Paragraph(f"Research Area {i}: {self._sanitize_text(query)}", 
                                                self.styles['SubHeader']))
                        elements.append(Paragraph(f"Sources Found: {sources_count}", 
                                                self.styles['CustomBodyText']))
                        
                        if content:
                            content_preview = content[:800] + "..." if len(content) > 800 else content
                            elements.append(Paragraph(self._sanitize_text(content_preview), 
                                                    self.styles['ResearchContent']))
                        
                        elements.append(Spacer(1, 10))
        else:
            elements.append(Paragraph("Research data collection encountered issues. Please refer to the appendix for execution details.", 
                                    self.styles['CustomBodyText']))
        
        return elements
    
    def _create_citations_section(self, results: Dict[str, Any]) -> List:
        elements = []
        
        elements.append(Paragraph("REFERENCES AND CITATIONS", self.styles['SectionHeader']))
        
        citation_results = results.get('citation_results', {})
        if citation_results.get('success') and citation_results.get('citations'):
            citations = citation_results['citations']
            
            elements.append(Paragraph(f"Total Citations Found: {len(citations)}", 
                                    self.styles['CustomBodyText']))
            elements.append(Spacer(1, 12))
            
            for i, citation in enumerate(citations, 1):
                citation_text = self._format_citation(citation, i)
                elements.append(Paragraph(self._sanitize_text(citation_text), 
                                        self.styles['Citation']))
                elements.append(Spacer(1, 4))
        else:
            elements.append(Paragraph("No citations were successfully gathered during this research.",
                                    self.styles['CustomBodyText']))
        
        if citation_results.get('search_queries'):
            elements.append(Spacer(1, 15))
            elements.append(Paragraph("Citation Search Queries", self.styles['SubHeader']))
            for query in citation_results['search_queries']:
                elements.append(Paragraph(f"• {self._sanitize_text(query)}", 
                                        self.styles['Citation']))
        
        return elements
    
    def _format_citation(self, citation, index):
        if hasattr(citation, 'title') and hasattr(citation, 'authors'):
            citation_text = f"{index}. {citation.authors} ({getattr(citation, 'year', 'N/A')}). {citation.title}."
            if hasattr(citation, 'journal') and citation.journal:
                citation_text += f" {citation.journal}."
            if hasattr(citation, 'url') and citation.url:
                citation_text += f" Available at: {citation.url}"
        elif hasattr(citation, 'to_apa'):
            citation_text = f"{index}. {citation.to_apa()}"
            if hasattr(citation, 'url') and citation.url:
                citation_text += f" Available at: {citation.url}"
        elif isinstance(citation, dict):
            title = citation.get('title', 'Unknown Title')
            authors = citation.get('authors', 'Unknown Authors')
            year = citation.get('year', 'N/A')
            citation_text = f"{index}. {authors} ({year}). {title}."
            if citation.get('url'):
                citation_text += f" Available at: {citation['url']}"
        else:
            citation_text = f"{index}. {str(citation)}"
        
        return citation_text
    
    def _create_appendix(self, results: Dict[str, Any]) -> List:
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("APPENDIX", self.styles['SectionHeader']))
        
        exec_summary = results.get('execution_summary', {})
        if exec_summary.get('execution_log'):
            elements.append(Paragraph("Execution Log", self.styles['SubHeader']))
            
            log_entries = exec_summary['execution_log'][-10:]
            for entry in log_entries:
                elements.append(Paragraph(self._sanitize_text(entry), self.styles['Citation']))
        
        if exec_summary.get('errors'):
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Errors Encountered", self.styles['SubHeader']))
            
            for error in exec_summary['errors']:
                elements.append(Paragraph(f"• {self._sanitize_text(error)}", 
                                        self.styles['Citation']))
        
        research_results = results.get('research_results', {})
        if research_results.get('execution_details'):
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Research Execution Details", self.styles['SubHeader']))
            
            details = research_results['execution_details']
            for key, value in details.items():
                elements.append(Paragraph(f"{key}: {self._sanitize_text(str(value))}", 
                                        self.styles['Citation']))
        
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Technical Specifications", self.styles['SubHeader']))
        
        tech_specs = f"""
        Research Framework: LangGraph Multi-Agent System<br/>
        Agents Employed: Planning, Research, Citation<br/>
        Execution Mode: Asynchronous Multi-threaded<br/>
        Quality Validation: Automated Assessment<br/>
        Report Format: PDF with structured sections<br/>
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        elements.append(Paragraph(tech_specs, self.styles['CustomBodyText']))
        
        return elements

class DeepResearchWorkflow:
    def __init__(self):
        self.planner = AgriculturalPlanningAgent()
        self.subsearch_agent = EnhancedSubsearchAgent(max_workers=4)
        self.citation_agent = EnhancedCitationAgent()
        self.pdf_generator = PDFReportGenerator()
        
        self.graph = self._build_workflow_graph()
        self.app = self.graph.compile(checkpointer=MemorySaver())

    def _build_workflow_graph(self) -> StateGraph:
        workflow = StateGraph(ResearchState)
        
        workflow.add_node("planning", self._planning_phase)
        workflow.add_node("research_execution", self._research_execution_phase)
        workflow.add_node("citation_gathering", self._citation_gathering_phase)
        workflow.add_node("quality_validation", self._quality_validation_phase)
        
        workflow.set_entry_point("planning")
        
        workflow.add_edge("planning", "research_execution")
        workflow.add_edge("research_execution", "citation_gathering")
        workflow.add_edge("citation_gathering", "quality_validation")
        workflow.add_edge("quality_validation", END)
        
        return workflow

    def _log_phase(self, state: ResearchState, phase: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {phase.upper()}: {message}"
        state.execution_log.append(log_entry)
        logger.info(log_entry)

    def _planning_phase(self, state: ResearchState) -> ResearchState:
        self._log_phase(state, "planning", "Starting research planning phase")
        state.current_phase = "planning"
        
        print(f"\n{'='*80}")
        print(f"PLANNING PHASE - Generating research plan")
        print(f"Title: {state.title}")
        print(f"Objective: {state.objective}")
        print(f"{'='*80}")
        
        try:
            planning_start = datetime.now()
            plan = self.planner.create_plan(state.title, state.objective)
            planning_time = (datetime.now() - planning_start).total_seconds()
            
            state.plan = plan
            state.tasks = plan.tasks if hasattr(plan, 'tasks') else []
            
            print(f"✓ Planning completed in {planning_time:.2f}s")
            print(f"  Generated {len(state.tasks)} research tasks")
            
            agent_assignments = getattr(plan, 'agent_assignments', [])
            print(f"  Agent assignments: {len(agent_assignments)} specialists")
            
            if state.tasks:
                print(f"\n📋 Research tasks created:")
                print(f"{'-'*60}")
                for i, task in enumerate(state.tasks[:5], 1):
                    task_desc = getattr(task, 'description', str(task))[:70]
                    if len(task_desc) > 70:
                        task_desc += "..."
                    print(f"{i}. {task_desc}")
                
                if len(state.tasks) > 5:
                    print(f"... and {len(state.tasks) - 5} more tasks")
            
            self._log_phase(state, "planning", f"Generated {len(state.tasks)} research tasks")
            self._log_phase(state, "planning", f"Agent assignments: {len(agent_assignments)} specialists")
            
        except Exception as e:
            error_msg = f"Planning phase failed: {str(e)}"
            state.errors.append(error_msg)
            self._log_phase(state, "planning", error_msg)
            print(f"✗ Planning failed: {str(e)}")
        
        return state

    def _research_execution_phase(self, state: ResearchState) -> ResearchState:
        self._log_phase(state, "research", "Starting research execution phase")
        state.current_phase = "research_execution"
        
        try:
            if not state.tasks:
                self._log_phase(state, "research", "No tasks available, using objective for research")
                research_queries = [f"{state.title} {state.objective}"]
            else:
                research_queries = []
                for task in state.tasks:
                    if hasattr(task, 'subsearch_queries') and task.subsearch_queries:
                        research_queries.extend(task.subsearch_queries[:2])
                    elif hasattr(task, 'description'):
                        research_queries.append(task.description)
                
                if not research_queries:
                    research_queries = [f"{state.title} {state.objective}"]
            
            research_queries = research_queries[:8]
            
            print(f"\n{'='*80}")
            print(f"RESEARCH EXECUTION - Processing {len(research_queries)} queries")
            print(f"{'='*80}")
            
            research_results = self._execute_research_with_display(research_queries)
            
            state.research_results = research_results
            
            if research_results.get('success'):
                self._log_phase(state, "research", f"Research completed: {research_results.get('successful_searches', 0)} successful queries")
                print(f"\n✓ Research phase completed successfully!")
                print(f"  Total sources found: {research_results.get('total_sources_found', 0)}")
                print(f"  Execution time: {research_results.get('total_execution_time', 0):.2f}s")
            else:
                self._log_phase(state, "research", "Research execution completed with limited results")
                print(f"\n⚠ Research phase completed with limited results")
            
        except Exception as e:
            error_msg = f"Research execution failed: {str(e)}"
            state.errors.append(error_msg)
            self._log_phase(state, "research", error_msg)
            state.research_results = {"success": False, "error": str(e)}
            print(f"\n✗ Research phase failed: {str(e)}")
        
        return state

    def _execute_research_with_display(self, research_queries: List[str]) -> Dict[str, Any]:
        """Execute research queries with real-time terminal display of results"""
        try:
            results = {
                "success": False,
                "search_results": [],
                "combined_content": "",
                "successful_searches": 0,
                "queries_processed": len(research_queries),
                "total_sources_found": 0,
                "total_execution_time": 0,
                "execution_details": {}
            }
            
            start_time = datetime.now()
            all_content = []
            
            for i, query in enumerate(research_queries, 1):
                print(f"\n[Query {i}/{len(research_queries)}] Searching: {query}")
                print(f"{'-'*60}")
                
                query_start = datetime.now()
                
                try:
                    # Use the correct method name from SubsearchAgent
                    query_result = self.subsearch_agent.subsearch_single_query(query)
                    query_time = (datetime.now() - query_start).total_seconds()
                    
                    if query_result and query_result.get('success'):
                        sources_found = len(query_result.get('sources', []))
                        content = query_result.get('combined_content', '')
                        
                        print(f"✓ Success: Found {sources_found} sources in {query_time:.2f}s")
                        
                        if content:
                            content_preview = content[:300].replace('\n', ' ')
                            if len(content) > 300:
                                content_preview += "..."
                            print(f"📄 Content preview: {content_preview}")
                            all_content.append(content)
                        
                        results["search_results"].append({
                            "query": query,
                            "success": True,
                            "sources_found": sources_found,
                            "combined_content": content,
                            "execution_time": query_time,
                            "sources": query_result.get('sources', [])
                        })
                        
                        results["successful_searches"] += 1
                        results["total_sources_found"] += sources_found
                        
                        if query_result.get('sources'):
                            print(f"🔗 Top sources:")
                            for j, source in enumerate(query_result['sources'][:3], 1):
                                if isinstance(source, dict):
                                    title = source.get('title', 'Unknown Title')[:50]
                                    url = source.get('url', 'No URL')[:60]
                                    print(f"   {j}. {title} - {url}")
                    else:
                        print(f"✗ Failed: No results found in {query_time:.2f}s")
                        results["search_results"].append({
                            "query": query,
                            "success": False,
                            "error": query_result.get('error', 'Unknown error') if query_result else 'No response from search agent',
                            "execution_time": query_time
                        })
                
                except Exception as e:
                    query_time = (datetime.now() - query_start).total_seconds()
                    print(f"✗ Error: {str(e)} (took {query_time:.2f}s)")
                    results["search_results"].append({
                        "query": query,
                        "success": False,
                        "error": str(e),
                        "execution_time": query_time
                    })
            
            total_time = (datetime.now() - start_time).total_seconds()
            results["total_execution_time"] = total_time
            results["combined_content"] = "\n\n".join(all_content)
            results["success"] = results["successful_searches"] > 0
            
            print(f"\n{'='*60}")
            print(f"RESEARCH SUMMARY:")
            print(f"  Successful queries: {results['successful_searches']}/{results['queries_processed']}")
            print(f"  Total sources: {results['total_sources_found']}")
            print(f"  Content length: {len(results['combined_content'])} characters")
            print(f"  Total time: {total_time:.2f}s")
            print(f"{'='*60}")
            
            return results
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "search_results": [],
                "combined_content": "",
                "successful_searches": 0,
                "queries_processed": len(research_queries),
                "total_sources_found": 0,
                "total_execution_time": 0
            }

    def _citation_gathering_phase(self, state: ResearchState) -> ResearchState:
        self._log_phase(state, "citations", "Starting citation gathering phase")
        state.current_phase = "citation_gathering"
        
        try:
            search_topic = f"{state.title} {state.objective}"
            if state.focus_areas:
                search_topic += f" {' '.join(state.focus_areas)}"
            
            num_citations = 20
            
            print(f"\n{'='*80}")
            print(f"CITATION GATHERING - Searching for {num_citations} citations")
            print(f"Topic: {search_topic}")
            print(f"{'='*80}")
            
            citation_start = datetime.now()
            citation_results = self.citation_agent.find_citations_basic(
                topic=search_topic,
                num_citations=num_citations
            )
            citation_time = (datetime.now() - citation_start).total_seconds()
            
            state.citation_results = citation_results
            
            valid_citations = citation_results.get('valid_count', 0) if citation_results and citation_results.get('success') else 0
            total_found = citation_results.get('total_found', 0) if citation_results else 0
            
            if citation_results and citation_results.get('success'):
                print(f"✓ Citation search completed in {citation_time:.2f}s")
                print(f"  Valid citations: {valid_citations}")
                print(f"  Total examined: {total_found}")
                
                if citation_results.get('citations') and len(citation_results['citations']) > 0:
                    print(f"\n📚 Sample citations found:")
                    print(f"{'-'*60}")
                    for i, citation in enumerate(citation_results['citations'][:5], 1):
                        citation_preview = self._get_citation_preview(citation)
                        print(f"{i}. {citation_preview}")
                    
                    if len(citation_results['citations']) > 5:
                        print(f"... and {len(citation_results['citations']) - 5} more")
                
                print(f"✓ Citation gathering successful!")
            else:
                print(f"⚠ Citation search completed with limited results in {citation_time:.2f}s")
                print(f"  Valid citations: {valid_citations}")
                
            self._log_phase(state, "citations", f"Citation search completed: {valid_citations} valid citations found")
            
        except Exception as e:
            error_msg = f"Citation gathering failed: {str(e)}"
            state.errors.append(error_msg)
            self._log_phase(state, "citations", error_msg)
            state.citation_results = {"success": False, "error": str(e)}
            print(f"✗ Citation gathering failed: {str(e)}")
        
        return state
    
    def _get_citation_preview(self, citation) -> str:
        """Get a short preview of a citation for display"""
        try:
            if hasattr(citation, 'title') and hasattr(citation, 'authors'):
                authors = citation.authors[:50] + "..." if len(citation.authors) > 50 else citation.authors
                title = citation.title[:60] + "..." if len(citation.title) > 60 else citation.title
                year = getattr(citation, 'year', 'N/A')
                return f"{authors} ({year}). {title}"
            elif hasattr(citation, 'to_apa'):
                apa = citation.to_apa()
                return apa[:80] + "..." if len(apa) > 80 else apa
            elif isinstance(citation, dict):
                authors = citation.get('authors', 'Unknown')[:30]
                title = citation.get('title', 'Unknown Title')[:40]
                year = citation.get('year', 'N/A')
                return f"{authors} ({year}). {title}"
            else:
                citation_str = str(citation)
                return citation_str[:80] + "..." if len(citation_str) > 80 else citation_str
        except:
            return str(citation)[:80]

    def _quality_validation_phase(self, state: ResearchState) -> ResearchState:
        self._log_phase(state, "validation", "Starting quality validation phase")
        state.current_phase = "quality_validation"
        
        print(f"\n{'='*80}")
        print(f"QUALITY VALIDATION - Assessing research quality")
        print(f"{'='*80}")
        
        try:
            validation_start = datetime.now()
            validation_results = self._validate_research_quality(state)
            validation_time = (datetime.now() - validation_start).total_seconds()
            
            state.execution_log.append("=== QUALITY VALIDATION RESULTS ===")
            for metric, value in validation_results.items():
                state.execution_log.append(f"{metric}: {value}")
            
            print(f"✓ Quality validation completed in {validation_time:.2f}s")
            print(f"\n📊 Quality Assessment:")
            print(f"{'-'*60}")
            print(f"Overall Quality: {validation_results.get('overall_quality', 'unknown').upper()}")
            print(f"Quality Score: {validation_results.get('quality_score', 0)}/100")
            print(f"Plan Generated: {'Yes' if validation_results.get('plan_generated') else 'No'}")
            print(f"Tasks Created: {validation_results.get('tasks_created', 0)}")
            print(f"Research Successful: {'Yes' if validation_results.get('research_successful') else 'No'}")
            print(f"Citations Found: {validation_results.get('citations_found', 0)}")
            print(f"Sources Found: {validation_results.get('sources_found', 0)}")
            print(f"Errors Encountered: {validation_results.get('errors_encountered', 0)}")
            print(f"Phases Completed: {validation_results.get('execution_phases_completed', 0)}/4")
            
            self._log_phase(state, "validation", f"Quality validation completed: {validation_results.get('overall_quality', 'unknown')}")
            
        except Exception as e:
            error_msg = f"Quality validation failed: {str(e)}"
            state.errors.append(error_msg)
            self._log_phase(state, "validation", error_msg)
            print(f"✗ Quality validation failed: {str(e)}")
        
        return state

    def _validate_research_quality(self, state: ResearchState) -> Dict[str, Any]:
        validation = {
            "plan_generated": state.plan is not None,
            "tasks_created": len(state.tasks) if state.tasks else 0,
            "research_successful": state.research_results.get('success', False) if state.research_results else False,
            "citations_found": state.citation_results.get('valid_count', 0) if state.citation_results else 0,
            "errors_encountered": len(state.errors),
            "execution_phases_completed": len([log for log in state.execution_log if "Starting" in log])
        }
        
        if state.research_results:
            validation.update({
                "sources_found": state.research_results.get('total_sources_found', 0),
                "successful_searches": state.research_results.get('successful_searches', 0),
                "research_execution_time": state.research_results.get('total_execution_time', 0)
            })
        
        quality_score = 0
        if validation["plan_generated"]: quality_score += 25
        if validation["tasks_created"] >= 5: quality_score += 25
        if validation["research_successful"]: quality_score += 30
        if validation["citations_found"] >= 5: quality_score += 20
        
        if validation["errors_encountered"] == 0:
            quality_score = min(100, quality_score)
        else:
            quality_score = max(0, quality_score - (validation["errors_encountered"] * 10))
        
        validation["quality_score"] = quality_score
        
        if quality_score >= 90:
            validation["overall_quality"] = "excellent"
        elif quality_score >= 75:
            validation["overall_quality"] = "good"
        elif quality_score >= 60:
            validation["overall_quality"] = "satisfactory"
        else:
            validation["overall_quality"] = "needs_improvement"
        
        return validation

    def _generate_title_from_objective(self, objective: str) -> str:
        objective_lower = objective.lower()
        
        if any(word in objective_lower for word in ['improve', 'increase', 'enhance', 'optimize']):
            if any(word in objective_lower for word in ['yield', 'production', 'harvest']):
                return f"Agricultural Yield Enhancement Study: {objective}"
            elif any(word in objective_lower for word in ['soil', 'fertility', 'nutrient']):
                return f"Soil Management Optimization Research: {objective}"
            else:
                return f"Agricultural Improvement Analysis: {objective}"
        
        elif any(word in objective_lower for word in ['evaluate', 'assess', 'analyze', 'study']):
            if any(word in objective_lower for word in ['technology', 'innovation', 'system']):
                return f"Agricultural Technology Assessment: {objective}"
            elif any(word in objective_lower for word in ['sustainability', 'environment', 'climate']):
                return f"Sustainable Agriculture Research: {objective}"
            else:
                return f"Agricultural Research Study: {objective}"
        
        elif any(word in objective_lower for word in ['develop', 'design', 'create']):
            return f"Agricultural Development Research: {objective}"
        
        elif any(word in objective_lower for word in ['compare', 'contrast']):
            return f"Comparative Agricultural Study: {objective}"
        
        else:
            return f"Agricultural Research: {objective}"

    def execute_research(self, objective: str, location: str = "Global", 
                        focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        title = self._generate_title_from_objective(objective)
        logger.info(f"Starting deep research workflow: {title}")
        
        initial_state = ResearchState(
            title=title,
            objective=objective,
            location=location,
            focus_areas=focus_areas or []
        )
        
        try:
            config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
            
            final_state = initial_state
            
            for state_update in self.app.stream(initial_state, config):
                if isinstance(state_update, dict):
                    for key, state_obj in state_update.items():
                        if hasattr(state_obj, "title") and hasattr(state_obj, "objective"):
                            final_state = state_obj
                            break
                elif hasattr(state_update, "title") and hasattr(state_update, "objective"):
                    final_state = state_update
                
                current_phase = getattr(final_state, 'current_phase', 'unknown')
                logger.info(f"Completed phase: {current_phase}")
                    
            return self._format_final_results(final_state)
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "title": title,
                "objective": objective,
                "location": location,
                "focus_areas": focus_areas or [],
                "planning_results": {},
                "research_results": {},
                "citation_results": {},
                "execution_summary": {"errors": [str(e)]},
                "quality_validation": None
            }

    def _format_final_results(self, state: ResearchState) -> Dict[str, Any]:
        agent_assignments = 0
        if state.plan and hasattr(state.plan, 'agent_assignments'):
            agent_assignments = len(state.plan.agent_assignments)
        
        return {
            "success": len(state.errors) == 0,
            "title": state.title,
            "objective": state.objective,
            "location": state.location,
            "focus_areas": state.focus_areas,
            
            "planning_results": {
                "plan_generated": state.plan is not None,
                "tasks_created": len(state.tasks) if state.tasks else 0,
                "agent_assignments": agent_assignments
            },
            
            "research_results": state.research_results or {},
            "citation_results": state.citation_results or {},
            
            "execution_summary": {
                "phases_completed": len([log for log in state.execution_log if "Starting" in log]),
                "errors_encountered": len(state.errors),
                "execution_log": state.execution_log,
                "errors": state.errors
            },
            
            "quality_validation": self._validate_research_quality(state) if state.current_phase == "quality_validation" else None
        }

    def save_pdf_report(self, results: Dict[str, Any], custom_filename: str = None) -> str:
        try:
            os.makedirs('Reports', exist_ok=True)
            
            if custom_filename:
                filename = custom_filename
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            else:
                safe_objective = results.get('objective', 'research').replace(' ', '_').replace('/', '_')[:30]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                filename = f"agricultural_research_{safe_objective}_{timestamp}.pdf"
            
            if not filename.startswith('Reports/'):
                filename = f"Reports/{filename}"
            
            self.pdf_generator.generate_pdf_report(results, filename)
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save PDF report: {str(e)}")
            raise e

    def display_results(self, results: Dict[str, Any]):
        print(f"\nDEEP RESEARCH WORKFLOW RESULTS")
        print("=" * 80)
        print(f"Title: {results['title']}")
        print(f"Objective: {results['objective']}")
        print(f"Location: {results['location']}")
        if results.get('focus_areas'):
            print(f"Focus Areas: {', '.join(results['focus_areas'])}")
        
        print(f"\nWorkflow Success: {'Yes' if results['success'] else 'No'}")
        
        exec_summary = results.get('execution_summary', {})
        print(f"\nEXECUTION SUMMARY")
        print(f"Phases Completed: {exec_summary.get('phases_completed', 0)}/4")
        print(f"Errors Encountered: {exec_summary.get('errors_encountered', 0)}")
        
        planning = results.get('planning_results', {})
        print(f"\nPLANNING PHASE")
        print(f"Plan Generated: {'Yes' if planning.get('plan_generated') else 'No'}")
        print(f"Tasks Created: {planning.get('tasks_created', 0)}")
        print(f"Agent Assignments: {planning.get('agent_assignments', 0)}")
        
        research = results.get('research_results', {})
        if research:
            print(f"\nRESEARCH PHASE")
            print(f"Research Success: {'Yes' if research.get('success') else 'No'}")
            print(f"Successful Searches: {research.get('successful_searches', 0)}/{research.get('queries_processed', 0)}")
            print(f"Sources Found: {research.get('total_sources_found', 0)}")
            print(f"Execution Time: {research.get('total_execution_time', 0):.2f}s")
        
        citations = results.get('citation_results', {})
        if citations:
            print(f"\nCITATION PHASE")
            print(f"Citation Success: {'Yes' if citations.get('success') else 'No'}")
            print(f"Valid Citations: {citations.get('valid_count', 0)}")
            print(f"Total Sources Examined: {citations.get('total_found', 0)}")
            
            if citations.get('success') and citations.get('citations'):
                print(f"\nCITATIONS FOUND:")
                print("-" * 60)
                for i, citation in enumerate(citations['citations'][:10], 1):
                    citation_text = self._format_citation_display(citation, i)
                    print(citation_text)
                    print()
                
                if len(citations['citations']) > 10:
                    print(f"... and {len(citations['citations']) - 10} more citations (see PDF report for complete list)")
        
        quality = results.get('quality_validation', {})
        if quality:
            print(f"\nQUALITY VALIDATION")
            print(f"Overall Quality: {quality.get('overall_quality', 'unknown').upper()}")
            print(f"Quality Score: {quality.get('quality_score', 0)}/100")
        
        if exec_summary.get('errors'):
            print(f"\nERRORS ENCOUNTERED")
            for error in exec_summary['errors']:
                print(f"- {error}")
        
        research_preview = research.get('combined_content', '') if research else ''
        if research_preview:
            print(f"\nRESEARCH PREVIEW")
            print("-" * 80)
            preview = research_preview[:1000]
            if len(research_preview) > 1000:
                preview += "\n\n... [Full research available in PDF] ..."
            print(preview)

    def _format_citation_display(self, citation, index):
        if hasattr(citation, 'title') and hasattr(citation, 'authors'):
            year = getattr(citation, 'year', 'N/A')
            citation_text = f"{index}. {citation.authors} ({year}). {citation.title}"
            if hasattr(citation, 'journal') and citation.journal:
                citation_text += f". {citation.journal}"
            if hasattr(citation, 'url') and citation.url:
                citation_text += f"\n   URL: {citation.url}"
        elif hasattr(citation, 'to_apa'):
            citation_text = f"{index}. {citation.to_apa()}"
            if hasattr(citation, 'url') and citation.url:
                citation_text += f"\n   URL: {citation.url}"
        elif isinstance(citation, dict):
            title = citation.get('title', 'Unknown Title')
            authors = citation.get('authors', 'Unknown Authors')
            year = citation.get('year', 'N/A')
            citation_text = f"{index}. {authors} ({year}). {title}"
            if citation.get('journal'):
                citation_text += f". {citation['journal']}"
            if citation.get('url'):
                citation_text += f"\n   URL: {citation['url']}"
        else:
            citation_text = f"{index}. {str(citation)}"
        
        return citation_text


def main():
    print("Deep Agricultural Research Pipeline")
    print("=" * 50)
    
    try:
        import reportlab
        print("PDF generation support available")
    except ImportError:
        print("ReportLab not found. Installing...")
        print("Run: pip install reportlab")
        sys.exit(1)
    
    workflow = DeepResearchWorkflow()
    
    objective = input("Research Objective: ").strip()
    if not objective:
        objective = "rice cultivation scope in North east India"
        print(f"Using default objective: {objective}")
    
    location = input("Location (optional): ").strip() or "Global"
    
    focus_input = input("Focus Areas (comma-separated, optional): ").strip()
    focus_areas = [area.strip() for area in focus_input.split(",")] if focus_input else []
    
    generated_title = workflow._generate_title_from_objective(objective)
    print(f"\nGenerated Title: {generated_title}")
    
    print(f"\nStarting deep research workflow...")
    print(f"This will involve: Planning -> Research -> Citations -> Quality Validation")
    print("Please wait, this may take several minutes...\n")
    
    results = workflow.execute_research(
        objective=objective,
        location=location,
        focus_areas=focus_areas
    )
    
    workflow.display_results(results)
    
    save_option = input(f"\nSave comprehensive PDF report? (y/n): ").strip().lower()
    if save_option == 'y':
        try:
            custom_name = input("Custom filename (optional, press Enter for auto-generated): ").strip()
            
            if custom_name:
                filename = workflow.save_pdf_report(results, custom_name)
            else:
                filename = workflow.save_pdf_report(results)
            
            print(f"PDF report saved successfully!")
            print(f"Location: {filename}")
            print(f"Report includes: Research findings, Citations, Execution log, and Quality metrics")
            
        except Exception as e:
            print(f"Failed to save PDF: {str(e)}")
            print("Make sure ReportLab is installed: pip install reportlab")
    
    json_backup = input(f"\nSave JSON backup for data analysis? (y/n): ").strip().lower()
    if json_backup == 'y':
        try:
            safe_objective = objective.replace(' ', '_').replace('/', '_')[:30]
            json_filename = f"Reports/research_data_{safe_objective}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            
            os.makedirs('Reports', exist_ok=True)
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"JSON backup saved to: {json_filename}")
        except Exception as e:
            print(f"Failed to save JSON backup: {str(e)}")

if __name__ == "__main__":
    main()
    