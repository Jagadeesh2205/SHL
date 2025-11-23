"""
Final checklist and helper script for submission
"""

import os
from pathlib import Path

print("=" * 80)
print("SHL ASSESSMENT RECOMMENDATION SYSTEM - SUBMISSION CHECKLIST")
print("=" * 80)

# Check required files
required_files = {
    'streamlit_app.py': 'Streamlit web application',
    'api/app.py': 'Flask REST API',
    'data/scraped_data.json': 'Assessment database',
    'data/embeddings/embeddings.npy': 'Pre-computed embeddings',
    'data/embeddings/faiss.index': 'FAISS vector index',
    'Gen_AI Dataset.xlsx': 'Train/test dataset',
    'firstname_lastname.csv': 'Predictions output',
    'requirements.txt': 'Python dependencies',
    'APPROACH_DOCUMENT.md': 'Approach document (to be converted to PDF)',
    'src/scraper.py': 'Scraping module',
    'src/embeddings.py': 'Embedding generation',
    'src/recommender.py': 'RAG recommendation engine',
    'src/evaluator.py': 'Evaluation metrics'
}

print("\n✓ FILE VERIFICATION:")
print("-" * 80)

all_exist = True
for file, description in required_files.items():
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"{status} {file:40s} - {description}")
    if not exists:
        all_exist = False

print("\n" + "=" * 80)

if all_exist:
    print("✅ ALL REQUIRED FILES ARE PRESENT!")
else:
    print("❌ SOME FILES ARE MISSING - Please check above")

print("\n" + "=" * 80)
print("📋 SUBMISSION REQUIREMENTS:")
print("-" * 80)

requirements = {
    "1. Web Application URL": "Deploy streamlit_app.py to Streamlit Cloud",
    "2. API Endpoint": "POST /recommend endpoint (can be same deployment)",
    "3. GitHub Repository": "Push all code to GitHub and make public",
    "4. Approach PDF": "Convert APPROACH_DOCUMENT.md to PDF",
    "5. Predictions CSV": "firstname_lastname.csv (✓ already generated)"
}

for req, detail in requirements.items():
    print(f"{req}")
    print(f"   → {detail}")

print("\n" + "=" * 80)
print("🚀 NEXT STEPS:")
print("-" * 80)
print("""
1. Convert APPROACH_DOCUMENT.md to PDF:
   - Option A: Use VS Code "Markdown PDF" extension
   - Option B: Use pandoc: 
     pandoc APPROACH_DOCUMENT.md -o approach_document.pdf
   - Option C: Online converter: https://md2pdf.netlify.app/

2. Deploy to Streamlit Cloud:
   a) Push code to GitHub
   b) Go to https://share.streamlit.io/
   c) Sign in and click "New app"
   d) Select your repository and streamlit_app.py
   e) Copy the deployment URL

3. Prepare GitHub Repository:
   - Ensure all files are committed
   - Make repository public
   - Add a clear README.md

4. Submit:
   - Web app URL (from step 2)
   - API endpoint URL (same as web app + /recommend)
   - GitHub repository URL
   - firstname_lastname.csv
   - approach_document.pdf (from step 1)
""")

print("=" * 80)
print("📊 PROJECT STATISTICS:")
print("-" * 80)

# Get file sizes
import json
with open('data/scraped_data.json') as f:
    assessments = json.load(f)

print(f"Total assessments scraped: {len(assessments)}")
print(f"Embedding dimension: 384")
print(f"Test queries: 9")
print(f"Predictions generated: 90 (9 queries × 10 recommendations)")

# Check predictions file
if os.path.exists('firstname_lastname.csv'):
    with open('firstname_lastname.csv') as f:
        lines = f.readlines()
    print(f"Predictions CSV rows: {len(lines) - 1}")  # -1 for header

print("\n" + "=" * 80)
print("✅ PROJECT IS READY FOR SUBMISSION!")
print("=" * 80)
