"""
Script: summarize_csv_files.py

Description:
Takes a directory of CSV files (with Q&A format), reads each file,
summarizes the text using the Google Gemini API, and writes an output CSV
with filename and summary.

Requirements:
pip install google-generativeai pandas

Usage:
python summarize_csv_files.py /path/to/csv/directory


"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# -------------------- CONFIGURATION --------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-flash" 
OUTPUT_FILE = "csv_summaries.csv"
# --------------------------------------------------------
def get_csv_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")]

def read_csv_text(file_path):
    try: df = pd.read_csv(file_path)
    except Exception as e:
        print(f"⚠ Error reading {file_path}: {e}")
        return ""

    all_text = []
    for col in df.columns:
        all_text.extend(df[col].dropna().astype(str).tolist())

    return "\n".join(all_text)

def summarize_text_with_gemini(text):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(f"Summarize the following Q&A data in 1 paragraph without any kind of formatting mainly about the contents (keep it technical and concise with clarity clearly mentioning what makes it distinct from other similar Q&A data):\n{text}")
        return response.text.strip()
    except Exception as e:
        print(f"⚠ Error summarizing text: {e}")
        return "Error generating summary"

def already_processed(filename):
    if not os.path.exists(OUTPUT_FILE): return False
    df = pd.read_csv(OUTPUT_FILE)
    return filename in df["filename"].values

def append_to_csv(filename, summary):
    new_row = pd.DataFrame([{"filename": filename, "summary": summary}])
    new_row.to_csv(OUTPUT_FILE, mode="a", header=not os.path.exists(OUTPUT_FILE), index=False)

def main(directory):
    genai.configure(api_key=API_KEY)
    csv_files = get_csv_files(directory)

    for csv_file in csv_files:
        fname = os.path.basename(csv_file)
        if already_processed(fname):
            print(f"⏩ Skipping (already processed): {fname}")
            continue

        print(f"📄 Processing: {fname}")
        text = read_csv_text(csv_file)
        summary = summarize_text_with_gemini(text) if text.strip() else "No text found"
        append_to_csv(fname, summary)
        print(f"✅ Added summary for {fname}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python summarize_csv_files_incremental.py /path/to/csv/directory")
        sys.exit(1)
    main(sys.argv[1])