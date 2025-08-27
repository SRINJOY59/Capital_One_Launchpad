from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import markdown
from bs4 import BeautifulSoup
from PyPDF2 import PdfMerger
import os

def markdown_to_pdf(md_content, output_file="output.pdf"):
    html_content = markdown.markdown(md_content)
    soup = BeautifulSoup(html_content, "html.parser")
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()

    custom_styles = {
        'CustomHeading1': ParagraphStyle(
            name="CustomHeading1", 
            parent=styles["Heading1"], 
            fontSize=16, 
            leading=20, 
            spaceAfter=12, 
            alignment=TA_CENTER
        ),
        'CustomHeading2': ParagraphStyle(
            name="CustomHeading2", 
            parent=styles["Heading2"], 
            fontSize=14, 
            leading=18, 
            spaceAfter=10
        ),
        'NormalBold': ParagraphStyle(
            name="NormalBold", 
            parent=styles["Normal"], 
            fontName="Helvetica-Bold"
        ),
        'NormalItalic': ParagraphStyle(
            name="NormalItalic", 
            parent=styles["Normal"], 
            fontName="Helvetica-Oblique"
        )
    }
    
    for style_name, style in custom_styles.items():
        styles.add(style)

    story = []

    for elem in soup.children:
        if elem.name == "h1":
            story.append(Paragraph(elem.get_text(), styles["CustomHeading1"]))
            story.append(Spacer(1, 12))
        elif elem.name == "h2":
            story.append(Paragraph(elem.get_text(), styles["CustomHeading2"]))
            story.append(Spacer(1, 10))
        elif elem.name == "h3":
            story.append(Paragraph(elem.get_text(), styles["Heading3"]))
            story.append(Spacer(1, 8))
        elif elem.name == "p":
            content = elem.decode_contents()
            if "<strong>" in content or "<b>" in content:
                content = content.replace("<strong>", "<b>").replace("</strong>", "</b>")
            story.append(Paragraph(content, styles["Normal"]))
            story.append(Spacer(1, 8))
        elif elem.name == "ul":
            items = []
            for li in elem.find_all("li", recursive=False):
                items.append(ListItem(Paragraph(li.get_text(), styles["Normal"])))
            story.append(ListFlowable(items, bulletType="bullet"))
            story.append(Spacer(1, 8))
        elif elem.name == "ol":
            items = []
            for li in elem.find_all("li", recursive=False):
                items.append(ListItem(Paragraph(li.get_text(), styles["Normal"])))
            story.append(ListFlowable(items, bulletType="1"))
            story.append(Spacer(1, 8))
        elif elem.name == "hr":
            story.append(Spacer(1, 12))
        else:
            if elem.string and elem.string.strip():
                story.append(Paragraph(elem.string.strip(), styles["Normal"]))
                story.append(Spacer(1, 6))

    doc.build(story)
    print(f"PDF generated: {output_file}")
    return output_file

def merge_pdfs(pdf_list, output_file="merged_output.pdf"):
    merger = PdfMerger()
    
    try:
        for pdf_file in pdf_list:
            if os.path.exists(pdf_file):
                merger.append(pdf_file)
                print(f"Added: {pdf_file}")
            else:
                print(f"Warning: {pdf_file} not found, skipping")
        
        with open(output_file, 'wb') as output:
            merger.write(output)
        
        merger.close()
        print(f"Merged PDF created: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error merging PDFs: {str(e)}")
        merger.close()
        return None

def create_report_from_markdown(title, sections, output_file="report.pdf"):
    markdown_content = f"# {title}\n\n"
    
    for section_title, section_content in sections.items():
        markdown_content += f"## {section_title}\n\n"
        markdown_content += f"{section_content}\n\n"
    
    return markdown_to_pdf(markdown_content, output_file)

def split_content_to_sections(content, max_length=2000):
    sections = {}
    words = content.split()
    current_section = 1
    current_content = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) > max_length and current_content:
            sections[f"Section {current_section}"] = " ".join(current_content)
            current_section += 1
            current_content = [word]
            current_length = len(word)
        else:
            current_content.append(word)
            current_length += len(word) + 1
    
    if current_content:
        sections[f"Section {current_section}"] = " ".join(current_content)
    
    return sections

def generate_comprehensive_report(title, objective, research_data, citations, output_file="comprehensive_report.pdf"):
    sections = {
        "Executive Summary": f"Research Objective: {objective}\n\nThis comprehensive report analyzes the research findings and provides actionable insights.",
        "Research Findings": research_data[:3000] if research_data else "Research data collection in progress.",
        "Methodology": "Multi-source data analysis approach utilizing academic research, practical guidelines, and expert knowledge.",
        "Key Insights": "Evidence-based conclusions derived from comprehensive analysis of available data sources.",
        "Recommendations": "Specific actionable guidance based on research findings and best practices.",
        "Implementation Strategy": "Practical framework for implementing recommendations with consideration of resources and constraints.",
        "Citations and References": citations[:2000] if citations else "Citations compiled from research sources."
    }
    
    return create_report_from_markdown(title, sections, output_file)

prompt = """
# AGRICULTURAL RESEARCH REPORT

## Executive Summary
**Status:** Research Completed  
**Analysis:** Comprehensive Data Review  

---

### Research Objective
**Description:** Analyze modern agricultural practices for sustainable farming  
**Methods:** Literature review, data analysis, expert consultation  
**Status:** Complete  

#### Key Findings:
- **Sustainability:** Organic methods show 15% higher soil retention
- **Technology:** Precision farming increases yield by 12-18%  
- **Cost Analysis:** Initial investment pays back within 3-4 seasons

### Recommendations:
- **Implementation:** Gradual transition to sustainable practices
- **Monitoring:** Regular soil health assessments
- **Training:** Farmer education programs essential
"""

if __name__ == "__main__":
    pdf1 = markdown_to_pdf(prompt, "Reports/report_part1.pdf")
    
    additional_content = """
# APPENDIX

## Technical Specifications
- **Framework:** Advanced agricultural analysis
- **Data Sources:** Multiple research databases
- **Quality Assurance:** Peer-reviewed methodologies

## Future Research
- Climate adaptation strategies
- Technology integration opportunities
- Economic impact studies
"""
    
    pdf2 = markdown_to_pdf(additional_content, "Reports/report_part2.pdf")

    merged_pdf = merge_pdfs([pdf1, pdf2], "Reports/final_agricultural_report.pdf")

    if merged_pdf:
        print(f"Complete report available: {merged_pdf}")
