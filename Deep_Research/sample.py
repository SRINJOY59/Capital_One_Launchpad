from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os

def export_py_to_pdf(py_files, output_pdf="output.pdf"):
    """
    Streams the contents of multiple .py files into a single PDF sequentially.

    Args:
        py_files (list): List of Python file paths to include.
        output_pdf (str): Output PDF filename.
    """
    # Prepare PDF document
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    for file_path in py_files:
        if os.path.exists(file_path):
            # Add filename as heading
            story.append(Paragraph(f"<b>{os.path.basename(file_path)}</b>", styles["Heading2"]))
            story.append(Spacer(1, 12))

            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Add code with monospaced formatting
            story.append(Preformatted(content, styles["Code"]))
            story.append(Spacer(1, 24))
        else:
            story.append(Paragraph(f"<b>{file_path} not found!</b>", styles["Normal"]))
            story.append(Spacer(1, 24))

    # Build PDF
    doc.build(story)
    print(f"✅ PDF successfully created: {output_pdf}")


if __name__ == "__main__":
    # Example usage
    files_to_export = ["citations_agent.py", "subsearchAgent.py"]
    export_py_to_pdf(files_to_export, "python_files_output.pdf")
