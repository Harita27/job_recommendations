import os
import fitz  # PyMuPDF
import spacy
import pandas as pd
import re
import requests
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load SBERT model once globally
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Comprehensive skill dictionary categorized by domain
SKILL_DICTIONARY = {
    "programming_languages": [
        "python", "java", "javascript", "c++", "c#", "ruby", "php", "swift", 
        "kotlin", "go", "rust", "typescript", "scala", "perl", "r"
    ],
    "web_development": [
        "html", "css", "react", "angular", "vue.js", "node.js", "django", 
        "flask", "express.js", "bootstrap", "jquery", "sass", "less", 
        "webpack", "gatsby", "next.js", "wordpress", "php", "laravel"
    ],
    "data_science": [
        "machine learning", "deep learning", "neural networks", "data analysis",
        "data visualization", "statistical analysis", "pandas", "numpy", 
        "scikit-learn", "tensorflow", "pytorch", "keras", "matplotlib", 
        "tableau", "power bi", "r", "spss", "sas", "computer vision", "nlp"
    ],
    "databases": [
        "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", 
        "sql server", "redis", "elasticsearch", "firebase", "dynamodb", 
        "cassandra", "neo4j", "graphql", "nosql"
    ],
    "devops": [
        "docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "gitlab ci", 
        "github actions", "terraform", "ansible", "puppet", "chef", "prometheus", 
        "grafana", "elk stack", "ci/cd", "linux", "bash", "shell scripting"
    ],
    "soft_skills": [
        "communication", "teamwork", "leadership", "problem solving", 
        "critical thinking", "time management", "project management", 
        "agile", "scrum", "kanban", "creativity", "adaptability"
    ]
}

# Flatten skill dictionary for faster lookups
ALL_SKILLS = set()
SKILL_CATEGORIES = {}
for category, skills in SKILL_DICTIONARY.items():
    for skill in skills:
        ALL_SKILLS.add(skill)
        SKILL_CATEGORIES[skill] = category

# Remotive API configuration
REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"

def fetch_remote_jobs(limit=100, category=None):
    """
    Fetch remote jobs from Remotive API
    
    Parameters:
    - limit: Maximum number of jobs to fetch
    - category: Optional job category filter (e.g., 'software-dev')
    
    Returns:
    - List of jobs from Remotive API
    """
    params = {'limit': limit}
    
    if category:
        params['category'] = category
    
    try:
        response = requests.get(REMOTIVE_API_URL, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        return data.get('jobs', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching jobs from Remotive API: {e}")
        return []

def extract_skills_from_job_posting(job_description):
    """
    Extract skills from job description text
    """
    # Clean and prepare the text
    cleaned_desc = job_description.lower()
    
    # Extract skills using our skills dictionary
    found_skills = []
    
    # First check for multi-word skills
    for skill in ALL_SKILLS:
        if ' ' in skill and skill in cleaned_desc:
            found_skills.append(skill)
    
    # Then check for single-word skills with word boundary check
    for skill in ALL_SKILLS:
        if ' ' not in skill:
            # Use regex with word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, cleaned_desc):
                found_skills.append(skill)
    
    return list(set(found_skills))  # Remove duplicates

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Enhanced skill extraction using NLP and pattern matching
def extract_skills(text):
    # Convert to lowercase for consistent matching
    text_lower = text.lower()
    
    # Clean the text - remove special characters and normalize whitespace
    cleaned_text = re.sub(r'[^\w\s]', ' ', text_lower)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Process with spaCy for context
    doc = nlp(cleaned_text)
    
    # Use NER and phrase matching
    extracted_skills = defaultdict(set)
    
    # Extract multi-word skills first (for phrases like "machine learning")
    for skill in ALL_SKILLS:
        if ' ' in skill and skill in text_lower:
            category = SKILL_CATEGORIES[skill]
            extracted_skills[category].add(skill)
    
    # Extract single-word skills and check for context
    for token in doc:
        if token.text.lower() in ALL_SKILLS:
            category = SKILL_CATEGORIES[token.text.lower()]
            extracted_skills[category].add(token.text.lower())
    
    # Convert to dictionary with lists
    result = {category: list(skills) for category, skills in extracted_skills.items()}
    
    # Add a "all" category with all skills combined
    all_identified_skills = []
    for skills in extracted_skills.values():
        all_identified_skills.extend(skills)
    
    result["all"] = sorted(all_identified_skills)
    
    return result

def semantic_skill_match(candidate_skills, job_skills_list):
    """
    Perform semantic matching between candidate skills and job skills
    using TF-IDF vectorization and cosine similarity
    """
    if not candidate_skills or not job_skills_list:
        return [], 0
    
    # Convert lists to space-separated strings for TF-IDF
    candidate_text = " ".join(candidate_skills)
    job_skills_text = [" ".join(job_skills_list)]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([candidate_text] + job_skills_text)
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Find direct skill matches
        exact_matches = [skill for skill in candidate_skills if skill in job_skills_list]
        
        # Calculate semantic similarity score (0-100)
        semantic_score = cosine_sim * 100
        
        return exact_matches, semantic_score
    except:
        # Fallback to direct matching if TF-IDF fails
        exact_matches = [skill for skill in candidate_skills if skill in job_skills_list]
        match_score = len(exact_matches) / len(job_skills_list) * 100 if job_skills_list else 0
        return exact_matches, match_score

def analyze_skill_gaps(extracted_skills, matched_jobs, top_n=5):
    """
    Analyze the skill gaps between candidate's resume and top matching jobs
    Returns recommended skills to learn
    """
    if not matched_jobs:
        return []
    
    # Get candidate's current skills
    candidate_skills = set(extracted_skills["all"])
    
    # Analyze top N matching jobs
    target_jobs = matched_jobs[:top_n] if len(matched_jobs) >= top_n else matched_jobs
    
    # Collect all required skills from top matching jobs
    required_skills = set()
    for job in target_jobs:
        job_skills = job.get('skills', '').lower().split(', ')
        required_skills.update([skill.strip() for skill in job_skills])
    
    # Find skills candidate is missing
    missing_skills = required_skills - candidate_skills
    
    # Count frequency of each missing skill in top jobs
    skill_frequency = {}
    for job in target_jobs:
        job_skills = [skill.strip().lower() for skill in job.get('skills', '').split(',')]
        for skill in missing_skills:
            if skill in job_skills:
                skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
    
    # Sort missing skills by frequency (most common first)
    recommended_skills = sorted(
        skill_frequency.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return recommended_skills

def get_ai_matches(resume_text, jobs_list):
    """
    Get AI-based job matches using sentence transformers
    """
    if not jobs_list:
        return []
        
    resume_embedding = sbert_model.encode(resume_text, convert_to_tensor=True)
    
    matched_jobs = []

    for job in jobs_list:
        job_title = job.get('title', '')
        job_desc = job.get('description', '')
        
        # Create a representative text from the job
        job_text = f"{job_title} {job_desc[:500]}"  # Limit description length

        job_embedding = sbert_model.encode(job_text, convert_to_tensor=True)

        similarity = util.cos_sim(resume_embedding, job_embedding).item() * 100  # percentage

        if similarity >= 40:  # Threshold for relevance
            matched_jobs.append({
                "title": job_title,
                "company": job.get('company_name', ''),
                "location": job.get('candidate_required_location', 'Remote'),
                "skills": ", ".join(extract_skills_from_job_posting(job_desc)),
                "matched_skills": "AI match",
                "match_score": round(similarity, 2),
                "experience": job.get('job_type', 'Not specified'),
                "salary": job.get('salary', 'Not specified'),
                "semantic_score": round(similarity, 2)
            })

    return matched_jobs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return "No file part"

    file = request.files['resume']
    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        resume_text = extract_text_from_pdf(file_path)
        extracted_skills = extract_skills(resume_text)
        
        # Fetch remote jobs from Remotive API
        # We'll fetch software development jobs by default
        remote_jobs = fetch_remote_jobs(limit=100, category='software-dev')
        
        if not remote_jobs:
            return "Error fetching job data from Remotive API"

        # 🔍 Skill-Based Matching
        skill_matched_jobs = []
        for job in remote_jobs:
            job_desc = job.get('description', '')
            job_title = job.get('title', '')
            company = job.get('company_name', '')
            location = job.get('candidate_required_location', 'Remote')
            
            # Extract skills from job description
            job_skills = extract_skills_from_job_posting(job_desc)
            job_skills_str = ", ".join(job_skills)
            
            # Match against candidate skills
            matched_skills = [skill for skill in extracted_skills["all"] if skill in job_skills]
            match_score = len(matched_skills) / len(job_skills) * 100 if job_skills else 0

            if matched_skills:
                skill_matched_jobs.append({
                    "title": job_title,
                    "company": company,
                    "location": location,
                    "skills": job_skills_str,
                    "matched_skills": matched_skills,
                    "match_score": round(match_score, 2),
                    "experience": job.get('job_type', 'Not specified'),
                    "salary": job.get('salary', 'Not specified'),
                    "source": "Skill Match"
                })

        # 🤖 AI-Based Matching
        ai_matched_jobs = get_ai_matches(resume_text, remote_jobs)
        for job in ai_matched_jobs:
            job["source"] = "AI Match"

        # 🧠 Merge & De-duplicate
        combined_jobs = { (job['title'], job['company']): job for job in skill_matched_jobs + ai_matched_jobs }
        final_jobs = list(combined_jobs.values())
        
        # Sort by match score (descending)
        final_jobs.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Analyze skill gaps based on top matching jobs
        skill_gaps = analyze_skill_gaps(extracted_skills, final_jobs)

        return render_template("result.html", 
                              extracted_skills=extracted_skills, 
                              jobs=final_jobs,
                              skill_gaps=skill_gaps,
                              categorized_skills=extracted_skills)

    return "Invalid file"

@app.route('/api/skills', methods=['POST'])
def api_extract_skills():
    if 'resume' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        resume_text = extract_text_from_pdf(file_path)
        extracted_skills = extract_skills(resume_text)

        return jsonify({"skills": extracted_skills})

    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    app.run(debug=True)