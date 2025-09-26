# 📌 Job Recommendations Web App

A Flask-based **AI-powered job recommendation system** that extracts skills from resumes and matches them with real-time job postings using both **rule-based skill matching** and **semantic AI matching** (via [SentenceTransformers](https://www.sbert.net/)).

---

## 🚀 Features

- 📂 Upload resume (**PDF/DOCX**)  
- 🧠 Extract candidate skills using **spaCy NLP** + a curated skill dictionary  
- 🔍 Fetch **real-time jobs** from [Remotive API](https://remotive.com/remote-jobs-api)  
- 🎯 Match jobs using:
  - **Skill-based matching** (direct overlaps with job descriptions)  
  - **AI-based semantic matching** (Sentence-BERT similarity)  
- 📊 Show job recommendations sorted by **match score**  
- 🛠 Analyze **skill gaps** and suggest missing skills  
- 🌐 Simple web interface (`index.html` & `result.html`)  

---

## 🛠 Tech Stack

- **Backend:** Python, Flask  
- **NLP:** spaCy, Sentence-BERT (`all-MiniLM-L6-v2`)  
- **ML Tools:** scikit-learn (TF-IDF, cosine similarity)  
- **Data Handling:** pandas, numpy  
- **File Processing:** PyMuPDF (resume parsing)  
- **Frontend:** HTML, Jinja2 templates  
- **API:** Remotive API (remote job listings)  

---

## 📂 Project Structure

job_recommendations/
├── app.py # Main Flask application
├── index.html # Homepage (resume upload form)
├── result.html # Results page (recommendations + skill gaps)
├── pyvenv.cfg # Virtual environment config (auto-generated)
├── uploads/ # Uploaded resumes (auto-created at runtime)
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation & Setup

### 1. Clone the repo
```bash
git clone https://github.com/Harita27/job_recommendations.git
cd job_recommendations
2. Create & activate virtual environment
bash
Copy code
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
3. Install dependencies
bash
Copy code
pip install flask spacy pandas requests scikit-learn numpy PyMuPDF sentence-transformers
4. Download spaCy model
bash
Copy code
python -m spacy download en_core_web_sm
▶️ Running the App
Start the server:

bash
Copy code
python app.py
Then open your browser at:

cpp
Copy code
http://127.0.0.1:5000
🖥️ Usage
Visit the homepage (index.html)

Upload your resume (PDF/DOCX)

The app will:

Extract your skills

Fetch jobs from Remotive API

Match jobs using both skill-based and AI-based approaches

Show recommended jobs with:

✅ Match score

🏢 Company name

📍 Location

💰 Salary info (if available)

Display missing skills you should learn

🔮 Future Improvements
Add support for DOCX parsing (currently only PDF handled in extract_text_from_pdf)

Store jobs & user data in a database (PostgreSQL / MongoDB)

Add authentication system for multiple users

Deploy on Azure / AWS / Heroku

Enhance UI/UX with CSS frameworks (Bootstrap/Tailwind)

Expand skill dictionary and add dynamic learning

📜 License
Open-source for learning and personal use.

🙌 Acknowledgements
spaCy for NLP

SentenceTransformers for semantic similarity

Remotive API for job listings

