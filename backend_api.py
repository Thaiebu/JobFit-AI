from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import requests
import json
import os
from datetime import datetime
import PyPDF2
import io

app = FastAPI(title="Job Search API", description="Search for jobs and match with resume")

# You need to get your API key from RapidAPI
RAPIDAPI_KEY =' your-rapidapi-key-here'  # Replace with your actual RapidAPI ke
RAPIDAPI_HOST = "jsearch.p.rapidapi.com"

class JobSearchRequest(BaseModel):
    query: str
    page: Optional[int] = 1
    num_pages: Optional[int] = 1
    country: Optional[str] = "ind"
    date_posted: Optional[str] = "today"
    employment_types: Optional[str] = None
    job_requirements: Optional[str] = None
    job_titles: Optional[str] = None
    company_types: Optional[str] = None
    employer: Optional[str] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    is_remote: Optional[bool] = None

class JobResponse(BaseModel):
    job_id: str
    job_title: str
    employer_name: str
    employer_logo: Optional[str]
    employer_website: Optional[str]
    job_publisher: str
    job_employment_type: str
    job_apply_link: str
    job_description: str
    job_is_remote: bool
    job_posted_at: str
    job_location: str
    job_city: Optional[str]
    job_state: Optional[str]
    job_country: str
    job_salary: Optional[str]
    job_min_salary: Optional[int]
    job_max_salary: Optional[int]
    job_salary_period: Optional[str]
    job_benefits: Optional[List[str]]

def get_job_search_headers():
    return {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF resume"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_txt(txt_file):
    """Extract text from uploaded text file"""
    try:
        content = txt_file.read()
        return content.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading text file: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Job Search API is running"}

@app.post("/search-jobs")
async def search_jobs(search_request: JobSearchRequest):
    """Search for jobs using JSearch API"""
    
    if not RAPIDAPI_KEY or RAPIDAPI_KEY == "your-rapidapi-key-here":
        raise HTTPException(
            status_code=500, 
            detail="Please set your RAPIDAPI_KEY environment variable"
        )
    
    url = "https://jsearch.p.rapidapi.com/search"
    
    # Prepare query parameters
    querystring = {
        "query": search_request.query,
        "page": str(search_request.page),
        "num_pages": str(search_request.num_pages),
        "country": search_request.country,
        "date_posted": search_request.date_posted
    }
    
    # Add optional parameters if provided
    if search_request.employment_types:
        querystring["employment_types"] = search_request.employment_types
    if search_request.job_requirements:
        querystring["job_requirements"] = search_request.job_requirements
    if search_request.job_titles:
        querystring["job_titles"] = search_request.job_titles
    if search_request.company_types:
        querystring["company_types"] = search_request.company_types
    if search_request.employer:
        querystring["employer"] = search_request.employer
    if search_request.salary_min:
        querystring["salary_min"] = str(search_request.salary_min)
    if search_request.salary_max:
        querystring["salary_max"] = str(search_request.salary_max)
    if search_request.is_remote is not None:
        querystring["remote_jobs_only"] = str(search_request.is_remote).lower()
    
    try:
        headers = get_job_search_headers()
        response = requests.get(url, headers=headers, params=querystring)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"API request failed: {response.text}"
            )
        
        data = response.json()
        
        # Format the response
        if 'data' in data and data['data']:
            jobs = []
            for job in data['data']:
                job_data = {
                    "job_id": job.get('job_id', ''),
                    "job_title": job.get('job_title', ''),
                    "employer_name": job.get('employer_name', ''),
                    "employer_logo": job.get('employer_logo'),
                    "employer_website": job.get('employer_website'),
                    "job_publisher": job.get('job_publisher', ''),
                    "job_employment_type": job.get('job_employment_type', ''),
                    "job_apply_link": job.get('job_apply_link', ''),
                    "job_description": job.get('job_description', ''),
                    "job_is_remote": job.get('job_is_remote', False),
                    "job_posted_at": job.get('job_posted_at', ''),
                    "job_location": job.get('job_location', ''),
                    "job_city": job.get('job_city'),
                    "job_state": job.get('job_state'),
                    "job_country": job.get('job_country', ''),
                    "job_salary": job.get('job_salary'),
                    "job_min_salary": job.get('job_min_salary'),
                    "job_max_salary": job.get('job_max_salary'),
                    "job_salary_period": job.get('job_salary_period'),
                    "job_benefits": job.get('job_benefits')
                }
                jobs.append(job_data)
            
            return {
                "status": "success",
                "total_jobs": len(jobs),
                "jobs": jobs,
                "search_parameters": search_request.dict()
            }
        else:
            return {
                "status": "success",
                "total_jobs": 0,
                "jobs": [],
                "message": "No jobs found for the given criteria"
            }
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API request error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/search-jobs-simple")
async def search_jobs_simple(
    query: str = Query(..., description="Job search query"),
    page: int = Query(1, description="Page number"),
    num_pages: int = Query(1, description="Number of pages"),
    country: str = Query("ind", description="Country code"),
    date_posted: str = Query("today", description="Date posted filter")
):
    """Simple job search endpoint with query parameters"""
    
    search_request = JobSearchRequest(
        query=query,
        page=page,
        num_pages=num_pages,
        country=country,
        date_posted=date_posted
    )
    
    return await search_jobs(search_request)

@app.post("/search-jobs-with-resume")
async def search_jobs_with_resume(
    query: str = Query(..., description="Job search query"),
    resume: UploadFile = File(..., description="Resume file (PDF or TXT)"),
    page: int = Query(1, description="Page number"),
    num_pages: int = Query(1, description="Number of pages"),
    country: str = Query("ind", description="Country code"),
    date_posted: str = Query("today", description="Date posted filter")
):
    """Search for jobs and match with uploaded resume"""
    
    # Validate file type
    allowed_types = ["application/pdf", "text/plain"]
    if resume.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported"
        )
    
    # Extract text from resume
    resume_text = ""
    try:
        if resume.content_type == "application/pdf":
            resume_text = extract_text_from_pdf(resume.file)
        elif resume.content_type == "text/plain":
            resume_text = extract_text_from_txt(resume.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing resume: {str(e)}")
    
    # Search for jobs
    search_request = JobSearchRequest(
        query=query,
        page=page,
        num_pages=num_pages,
        country=country,
        date_posted=date_posted
    )
    
    job_results = await search_jobs(search_request)
    
    # Simple keyword matching for demonstration
    # In a real application, you'd use more sophisticated NLP/ML techniques
    resume_keywords = set(resume_text.lower().split())
    
    if job_results["jobs"]:
        for job in job_results["jobs"]:
            job_keywords = set((job["job_description"] + " " + job["job_title"]).lower().split())
            matching_keywords = resume_keywords.intersection(job_keywords)
            match_score = len(matching_keywords) / len(job_keywords) if job_keywords else 0
            job["match_score"] = round(match_score * 100, 2)
            job["matching_keywords"] = list(matching_keywords)[:10]  # Limit to top 10
        
        # Sort jobs by match score
        job_results["jobs"].sort(key=lambda x: x["match_score"], reverse=True)
    
    return {
        **job_results,
        "resume_processed": True,
        "resume_length": len(resume_text),
        "message": "Jobs ranked by relevance to your resume"
    }

@app.get("/job-details/{job_id}")
async def get_job_details(job_id: str):
    """Get detailed information about a specific job"""
    
    if not RAPIDAPI_KEY or RAPIDAPI_KEY == "your-rapidapi-key-here":
        raise HTTPException(
            status_code=500, 
            detail="Please set your RAPIDAPI_KEY environment variable"
        )
    
    url = "https://jsearch.p.rapidapi.com/job-details"
    
    querystring = {"job_id": job_id}
    
    try:
        headers = get_job_search_headers()
        response = requests.get(url, headers=headers, params=querystring)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"API request failed: {response.text}"
            )
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API request error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_configured": RAPIDAPI_KEY != "your-rapidapi-key-here"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)