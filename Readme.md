# 💼 JobFit AI - Smart Job Search with Resume Matching

**JobFit AI** is an intelligent job recommendation web app built using **Streamlit**, which allows users to:

* 🔍 Search jobs by keyword, location, and filters
* 📋 Upload a resume (PDF or TXT)
* 🧠 Get **AI-powered similarity scores** between resume content and job descriptions
* ⭐ View company reviews, salary insights, benefits, and detailed job data

---

## 🚀 Features

* **Resume Upload & Matching**: Upload your resume and instantly see how well it matches with job listings using sentence embeddings and cosine similarity.
* **Real-Time Job Search**: Search for jobs using keywords, location, and filter by recency.
* **Job Insights**: View job descriptions, employment type, salary, benefits, and whether it's remote.
* **Employer Reviews**: See star ratings and links to external company reviews.
* **Similarity Emojis & Scores**: Quickly understand job relevance to your resume with visual match indicators.

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: FastAPI (for job search API)
* **AI Model**: Sentence Transformers (`all-MiniLM-L6-v2`)
* **Libraries**: `streamlit`, `requests`, `PyPDF2`, `scikit-learn`, `sentence-transformers`, `pandas`, `numpy`

---

## 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Thaiebu/JobFit-AI.git
   cd JobFit-AI
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start your FastAPI backend** (required):
    ```
    python backend_api.py
    ```
   Make sure your FastAPI server is running and accessible at the default base URL:

   ```
   http://localhost:8000
   ```

   If needed, adjust the `API_BASE_URL` in the script.

5. **Run the Streamlit app:**

   ```bash
   streamlit run job_with_resume_ui.py
   ```

---

## 📁 Project Structure

```
jobfit-ai/
│
├── job_with_resume_ui.py                 # Main Streamlit application
├── backend_api.py # Backend fast api
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## ✅ Requirements

* Python 3.8+
* Internet connection (for sentence-transformer model and API calls)

---

## 🧪 Example Usage

1. Run the app with `streamlit run job_with_resume_ui.py`.
2. Upload your resume in the left panel.
3. Enter a job search term like *"data scientist in Bangalore"*.
4. View jobs and similarity scores based on your resume content.
5. Click **Apply Now** or **View Reviews** for more job insights.

---

## 📸 Screenshots

| Resume Upload                                  | Job Results with Match Score                    |
| ---------------------------------------------- | ----------------------------------------------- |
| ![Upload](https://via.placeholder.com/300x180) | ![Results](https://via.placeholder.com/300x180) |

---

## 🤖 How Similarity Works

We use [Sentence Transformers](https://www.sbert.net/) to encode both your resume and each job description into vector representations, and then calculate the **cosine similarity** between them to estimate how closely your resume aligns with the job.

---

## 📬 Contact

Feel free to reach out if you'd like to contribute or suggest features:

**Author**: \[Mohamed Thaiebu]
**Email**: [thaiebu785@gmail.com](mailto:thaiebu785@gmail.com)
**LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---
