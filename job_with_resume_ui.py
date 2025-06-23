import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import re

# Set page config
st.set_page_config(
    page_title="JobFit AI - Job Search with Resume Matching",
   # page_description="Find jobs that match your resume with AI-powered similarity scoring and detailed job insights.",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .job-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .job-title {
        color: #1f77b4;
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .employer-name {
        color: #2e8b57;
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .job-meta {
        color: #666;
        font-size: 0.9em;
        margin: 5px 0;
    }
    
    .review-card {
        background-color: #f0f8ff;
        border: 1px solid #b0d4ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .review-header {
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    
    .star-rating {
        color: #ffd700;
        font-size: 1.2em;
    }
    
    .salary-info {
        background-color: #e8f5e8;
        padding: 8px;
        border-radius: 4px;
        font-weight: bold;
        color: #2e8b57;
    }
    
    .remote-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    
    .similarity-score {
        background: linear-gradient(90deg, #ff4757, #ffa502, #2ed573);
        color: white;
        padding: 8px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
        display: inline-block;
        margin: 5px 0;
    }
    
    .similarity-high {
        background: linear-gradient(90deg, #2ed573, #1dd1a1) !important;
    }
    
    .similarity-medium {
        background: linear-gradient(90deg, #ffa502, #ff9ff3) !important;
    }
    
    .similarity-low {
        background: linear-gradient(90deg, #ff4757, #ff3838) !important;
    }
    
    .resume-upload {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8fff8;
        margin: 20px 0;
    }
    
  .apply-button {
    background-color: #007BFF;       /* Bootstrap primary blue */
    color: white !important;
    padding: 10px 15px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 10px 0;
    cursor: pointer;
    border-radius: 4px;
    border: none;
    transition: background-color 0.3s ease;
}
.apply-button:hover {
    background-color: #0056b3;       /* Darker blue on hover */
}
</style>
""", unsafe_allow_html=True)

# API base URL - change this to your FastAPI server URL
API_BASE_URL = "http://localhost:8000"

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer model (cached for performance)"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading sentence transformer model: {str(e)}")
        return None

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def clean_text(text):
    """Clean and preprocess text for better similarity matching"""
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\-\.\,\;\:\!\?]', ' ', text)
    # Convert to lowercase
    text = text.lower().strip()
    return text

def calculate_similarity_score(resume_text, job_description, model):
    """Calculate similarity score between resume and job description"""
    try:
        if not resume_text or not job_description or not model:
            return 0.0
        
        # Clean texts
        clean_resume = clean_text(resume_text)
        clean_job = clean_text(job_description)
        
        if not clean_resume or not clean_job:
            return 0.0
        
        # Generate embeddings
        resume_embedding = model.encode([clean_resume])
        job_embedding = model.encode([clean_job])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        
        # Convert to percentage and round
        return round(float(similarity * 100), 1)
        
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return 0.0

def get_similarity_class(score):
    """Get CSS class based on similarity score"""
    if score >= 70:
        return "similarity-high"
    elif score >= 40:
        return "similarity-medium"
    else:
        return "similarity-low"

def get_similarity_emoji(score):
    """Get emoji based on similarity score"""
    if score >= 80:
        return "🔥"  # Hot match
    elif score >= 70:
        return "⭐"  # Great match
    elif score >= 50:
        return "👍"  # Good match
    elif score >= 30:
        return "👌"  # Okay match
    else:
        return "📋"  # Basic match

def search_jobs(query, page=1, country="ind", date_posted="today"):
    """Search for jobs using the FastAPI endpoint"""
    try:
        url = f"{API_BASE_URL}/search-jobs-simple"
        params = {
            "query": query,
            "page": page,
            "country": country,
            "date_posted": date_posted
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error searching jobs: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_job_details(job_id):
    """Get detailed job information and reviews"""
    try:
        url = f"{API_BASE_URL}/job-details/{job_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting job details: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def display_star_rating(score, max_score=5):
    """Display star rating"""
    filled_stars = int(score)
    half_star = 1 if (score - filled_stars) >= 0.5 else 0
    empty_stars = max_score - filled_stars - half_star
    
    stars = "⭐" * filled_stars + "⭐" * half_star + "☆" * empty_stars
    return f"{stars} ({score}/{max_score})"

def truncate_text(text, max_length=200):
    """Truncate text and add read more option"""
    if len(text) <= max_length:
        return text, False
    return text[:max_length] + "...", True

def display_job_card(job, job_index, resume_text=None, similarity_model=None):
    """Display a job card with all details and similarity score"""
    
    # Calculate similarity score if resume is provided
    similarity_score = 0.0
    if resume_text and similarity_model and job.get("job_description"):
        similarity_score = calculate_similarity_score(
            resume_text, 
            job.get("job_description", ""), 
            similarity_model
        )
    
    with st.container():
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Job title and employer with similarity score
            title_col, score_col = st.columns([3, 1])
            
            with title_col:
                st.markdown(f'<div class="job-title">{job.get("job_title", "N/A")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="employer-name">🏢 {job.get("employer_name", "N/A")}</div>', unsafe_allow_html=True)
            
            with score_col:
                if resume_text and similarity_score > 0:
                    similarity_class = get_similarity_class(similarity_score)
                    similarity_emoji = get_similarity_emoji(similarity_score)
                    st.markdown(f"""
                        <div class="similarity-score {similarity_class}">
                            {similarity_emoji} {similarity_score}% Match
                        </div>
                    """, unsafe_allow_html=True)
            
            # Job metadata
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            
            with col_meta1:
                st.markdown(f'<div class="job-meta">📍 {job.get("job_location", "N/A")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="job-meta">⏰ {job.get("job_posted_at", "N/A")}</div>', unsafe_allow_html=True)
            
            with col_meta2:
                st.markdown(f'<div class="job-meta">💼 {job.get("job_employment_type", "N/A")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="job-meta">📰 {job.get("job_publisher", "N/A")}</div>', unsafe_allow_html=True)
            
            with col_meta3:
                if job.get("job_is_remote"):
                    st.markdown('<span class="remote-badge">🏠 Remote</span>', unsafe_allow_html=True)
                
                if job.get("job_salary"):
                    st.markdown(f'<div class="salary-info">💰 {job.get("job_salary")}</div>', unsafe_allow_html=True)
                elif job.get("job_min_salary") and job.get("job_max_salary"):
                    salary_range = f"${job.get('job_min_salary'):,} - ${job.get('job_max_salary'):,}"
                    if job.get("job_salary_period"):
                        salary_range += f" {job.get('job_salary_period')}"
                    st.markdown(f'<div class="salary-info">💰 {salary_range}</div>', unsafe_allow_html=True)
        
        with col2:
            # Employer logo
            if job.get("employer_logo"):
                st.image(job.get("employer_logo"), width=100)
            
            # Apply button
            if job.get("job_apply_link"):
                st.markdown(f"""
                    <a href="{job.get('job_apply_link')}" target="_blank" class="apply-button">
                        🚀 Apply Now
                    </a>
                """, unsafe_allow_html=True)
        
        # Show similarity insights if available
        if resume_text and similarity_score > 0:
            with st.expander(f"🔍 Similarity Insights ({similarity_score}% match)", expanded=False):
                if similarity_score >= 70:
                    st.success(f"🎯 **Excellent Match!** This job aligns very well with your resume. Your skills and experience seem highly relevant.")
                elif similarity_score >= 50:
                    st.info(f"👍 **Good Match!** This job has good alignment with your background. Consider highlighting relevant skills in your application.")
                elif similarity_score >= 30:
                    st.warning(f"👌 **Moderate Match.** Some of your skills may be relevant. You might need to emphasize transferable skills.")
                else:
                    st.error(f"📋 **Low Match.** This job may require skills not prominent in your resume. Consider if you have relevant experience to highlight.")
        
        # Job description with read more
        if job.get("job_description"):
            description = job.get("job_description").strip()
            if description:
                truncated_desc, needs_expansion = truncate_text(description, 300)
                
                # Show truncated description
                st.markdown(f"**Job Description:**")
                st.markdown(f'<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">{truncated_desc}</div>', unsafe_allow_html=True)
                
                # Read more button
                if needs_expansion:
                    if st.button(f"📖 Read More", key=f"read_more_{job_index}"):
                        st.session_state[f"show_full_desc_{job_index}"] = True
                    
                    # Show full description if read more is clicked
                    if st.session_state.get(f"show_full_desc_{job_index}", False):
                        st.markdown("**Full Description:**")
                        st.markdown(f'<div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0;">{description}</div>', unsafe_allow_html=True)
                        
                        if st.button(f"📕 Show Less", key=f"show_less_{job_index}"):
                            st.session_state[f"show_full_desc_{job_index}"] = False
                            st.rerun()
        
        # Benefits
        if job.get("job_benefits"):
            st.markdown("**Benefits:**")
            for benefit in job.get("job_benefits", []):
                st.markdown(f"✅ {benefit}")
        
        # Review button
        review_button_col1, review_button_col2 = st.columns([1, 4])
        
        with review_button_col1:
            if st.button(f"⭐ View Reviews", key=f"review_btn_{job_index}"):
                st.session_state[f"show_reviews_{job_index}"] = True
        
        # Show reviews if button is clicked
        if st.session_state.get(f"show_reviews_{job_index}", False):
            with st.spinner("Loading reviews..."):
                job_details = get_job_details(job.get("job_id"))
                
                if job_details and job_details.get("data") and len(job_details["data"]) > 0:
                    employer_reviews = job_details["data"][0].get("employer_reviews", [])
                    
                    if employer_reviews:
                        st.markdown("### 📊 Employer Reviews")
                        
                        for review in employer_reviews:
                            st.markdown(f"""
                            <div class="review-card">
                                <div class="review-header">{review.get('publisher', 'Unknown')} Reviews</div>
                                <div class="star-rating">{display_star_rating(review.get('score', 0), review.get('max_score', 5))}</div>
                                <p><strong>Review Count:</strong> {review.get('review_count', 0)} reviews</p>
                                <a href="{review.get('reviews_link', '#')}" target="_blank">📖 Read Full Reviews</a>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No reviews available for this employer.")
                else:
                    st.error("Unable to load reviews at this time.")
            
            # Hide reviews button
            if st.button(f"🔼 Hide Reviews", key=f"hide_reviews_{job_index}"):
                st.session_state[f"show_reviews_{job_index}"] = False
                st.rerun()
        
        st.markdown("---")

def main():
    st.title("💼 AI powered Job Search Portal")
    st.markdown("Find your dream job with detailed information, company reviews, and AI-powered resume matching!")
    
    # Initialize session state
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = None
    if 'similarity_model' not in st.session_state:
        st.session_state.similarity_model = None
    if 'auto_search' not in st.session_state:
        st.session_state.auto_search = False
    
    # Resume upload section
    st.markdown("### 📋 Upload Your Resume for AI-Powered Job Matching")
    
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=['pdf', 'txt'],
        help="Upload your resume in PDF or TXT format to get personalized job matching scores"
    )
    
    # Process uploaded resume
    if uploaded_file is not None:
        with st.spinner("Processing your resume..."):
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = str(uploaded_file.read(), "utf-8")
            
            if resume_text:
                st.session_state.resume_text = resume_text
                
                # Load similarity model if not already loaded
                if st.session_state.similarity_model is None:
                    with st.spinner("Loading AI model for similarity analysis..."):
                        st.session_state.similarity_model = load_sentence_transformer()
                
                # Show resume preview
                with st.expander(f"📄 Resume Preview ({len(resume_text)} characters)", expanded=False):
                    st.text_area("Resume Content", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, height=200, disabled=True)
                
                st.success("✅ Resume uploaded successfully! Job similarity scores will now be displayed.")
            else:
                st.error("Could not extract text from the uploaded file. Please try a different file.")
    
    # Clear resume button
    if st.session_state.resume_text:
        if st.button("🗑️ Clear Resume"):
            st.session_state.resume_text = None
            st.rerun()
    
    # Sidebar for search parameters
    with st.sidebar:
        st.header("🔍 Search Filters")
        
        # Resume status
        if st.session_state.resume_text:
            st.success("✅ Resume uploaded - AI matching enabled!")
        else:
            st.info("Upload resume for personalized job matching")
        
        # Search query
        query = st.text_input(
            "Job Search Query",
            value="developer jobs in chennai",
            help="Enter job title, location, or keywords"
        )
        
        # Country selection
        country = st.selectbox(
            "Country",
            options=["ind", "us", "uk", "ca", "au"],
            help="Select country for job search"
        )
        
        # Date posted filter
        date_posted = st.selectbox(
            "Date Posted",
            options=["today", "3days", "week", "month", "all"],
            help="Filter jobs by posting date"
        )
        
        # Page number
        page = st.number_input(
            "Page Number",
            min_value=1,
            max_value=10,
            value=1,
            help="Page number for pagination"
        )
        
        # Sorting options when resume is uploaded
        if st.session_state.resume_text:
            sort_by = st.selectbox(
                "Sort Results By",
                options=["similarity", "date", "default"],
                help="Sort jobs by similarity score, posting date, or default order"
            )
        else:
            sort_by = "default"
        
        # Search button
        search_button = st.button("🔍 Search Jobs", type="primary")
    
    # Main content area
    if search_button or st.session_state.auto_search:
        st.session_state.auto_search = True
        
        with st.spinner("Searching for jobs..."):
            results = search_jobs(query, page, country, date_posted)
        
        if results and results.get("jobs"):
            jobs = results["jobs"]
            
            # Calculate similarity scores for all jobs if resume is uploaded
            if st.session_state.resume_text and st.session_state.similarity_model:
                with st.spinner("Calculating job similarity scores..."):
                    for job in jobs:
                        if job.get("job_description"):
                            similarity_score = calculate_similarity_score(
                                st.session_state.resume_text,
                                job.get("job_description", ""),
                                st.session_state.similarity_model
                            )
                            job['similarity_score'] = similarity_score
                        else:
                            job['similarity_score'] = 0.0
                
                # Sort jobs based on user preference
                if sort_by == "similarity":
                    jobs.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                elif sort_by == "date":
                    # This would require parsing dates, simplified here
                    pass  # Keep original order for now
            
            # Display search summary
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.success(f"Found {len(jobs)} jobs for '{query}'")
            
            with col2:
                if st.session_state.resume_text:
                    avg_score = sum(job.get('similarity_score', 0) for job in jobs) / len(jobs)
                    st.metric("Avg Match Score", f"{avg_score:.1f}%")
            
            with col3:
                if st.session_state.resume_text:
                    high_match_count = sum(1 for job in jobs if job.get('similarity_score', 0) >= 70)
                    st.metric("High Matches", high_match_count)
            
            # Display best matches summary if resume is uploaded
            if st.session_state.resume_text and jobs:
                best_matches = [job for job in jobs if job.get('similarity_score', 0) >= 70]
                if best_matches:
                    st.info(f"🎯 Found {len(best_matches)} high-similarity matches (70%+) based on your resume!")
                elif any(job.get('similarity_score', 0) >= 50 for job in jobs):
                    good_matches = [job for job in jobs if job.get('similarity_score', 0) >= 50]
                    st.info(f"👍 Found {len(good_matches)} good matches (50%+) based on your resume!")
            
            # Display jobs
            for i, job in enumerate(jobs):
                display_job_card(
                    job, 
                    i, 
                    resume_text=st.session_state.resume_text,
                    similarity_model=st.session_state.similarity_model
                )
            
            # Pagination info
            if len(jobs) > 0:
                st.info(f"Showing page {page} results. Use the sidebar to navigate to other pages.")
        
        elif results:
            st.warning("No jobs found for your search criteria. Try adjusting your search terms.")
        else:
            st.error("Unable to search for jobs. Please check your connection and try again.")
    
    else:
        # Welcome message
        st.info("👈 Upload your resume and use the sidebar to search for personalized job matches!")
        
        # Sample searches
        st.markdown("### 🌟 Popular Searches")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💻 Software Developer Jobs"):
                st.session_state["sample_query"] = "software developer jobs"
                st.rerun()
        
        with col2:
            if st.button("📊 Data Analyst Jobs"):
                st.session_state["sample_query"] = "data analyst jobs"
                st.rerun()
        
        with col3:
            if st.button("🎨 UI/UX Designer Jobs"):
                st.session_state["sample_query"] = "ui ux designer jobs"
                st.rerun()
        
        # Handle sample query selection
        if "sample_query" in st.session_state:
            st.session_state.auto_search = True
            query = st.session_state["sample_query"]
            del st.session_state["sample_query"]
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>💼 Job Search Portal | Built with Streamlit & AI-Powered Matching</p>
            <p>Find your next opportunity with detailed job information, company reviews, and personalized resume matching!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()