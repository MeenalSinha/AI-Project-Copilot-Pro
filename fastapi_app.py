"""
AI Multi-Purpose Copilot - FastAPI REST API
Powered by Google Gemini 2.0 Flash

REST API Endpoints for:
1. Career & Resume Mentor
2. YouTube & Content Creation Coach
3. Fitness & Diet Planning Assistant
4. Academic Research & Study Planner
5. Startup / Business Launch Consultant

Features:
- RESTful API endpoints
- Memory system with user sessions
- Export to multiple formats
- OpenAPI documentation
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.genai as genai

# Load environment
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "gemini-2.0-flash-001"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set. Add it to your .env file.")

# Initialize Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Data directories
DATA_DIR = "api_data"
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")

for directory in [DATA_DIR, SESSIONS_DIR, EXPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="AI Multi-Purpose Copilot API",
    description="RESTful API for 5 specialized AI copilots powered by Google Gemini",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserProfile(BaseModel):
    name: Optional[str] = None
    field: Optional[str] = None
    email: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = {}


class GeneratePlanRequest(BaseModel):
    copilot_type: str = Field(..., description="Type of copilot: career, youtube, fitness, academic, startup")
    goal: str = Field(..., description="User's goal or objective")
    user_profile: Optional[UserProfile] = None
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Model temperature for creativity")


class GeneratePlanResponse(BaseModel):
    plan_id: str
    copilot_type: str
    goal: str
    plan: str
    timestamp: str
    session_id: Optional[str] = None


class ExportRequest(BaseModel):
    plan_id: str
    format: str = Field(..., description="Export format: markdown, json, text")
    title: Optional[str] = None


class ProjectHistoryResponse(BaseModel):
    projects: List[Dict[str, Any]]
    total: int


class SessionResponse(BaseModel):
    session_id: str
    projects: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    created_at: str


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class SessionManager:
    """Manage user sessions and memory"""
    
    @staticmethod
    def create_session() -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "user_profile": {},
            "projects": []
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return session_id
    
    @staticmethod
    def get_session(session_id: str) -> Dict[str, Any]:
        """Get session data"""
        session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(session_file, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def update_session(session_id: str, data: Dict[str, Any]):
        """Update session data"""
        session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        
        with open(session_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def add_project_to_session(session_id: str, project: Dict[str, Any]):
        """Add a project to session"""
        session_data = SessionManager.get_session(session_id)
        session_data["projects"].append(project)
        SessionManager.update_session(session_id, session_data)


# ============================================================================
# COPILOT CONFIGURATIONS
# ============================================================================

COPILOT_PROMPTS = {
    "career": """You are an expert Career & Resume Mentor with 15+ years of experience.

User Profile: {user_profile}
User Goal: {goal}

Provide a comprehensive career plan with:
1. Career Analysis (current situation, strengths, recommendations)
2. Action Plan (immediate, short-term, long-term steps)
3. Resume & LinkedIn Strategy
4. Interview Preparation
5. Resources & Next Steps

Be specific, actionable, and encouraging.""",

    "youtube": """You are a YouTube & Content Creation Coach with expertise in video production and audience growth.

User Profile: {user_profile}
User Goal: {goal}

Create a comprehensive content strategy with:
1. Channel Strategy (niche, audience, positioning)
2. Content Plan (first 10 videos, content pillars, schedule)
3. Production Guidelines (equipment, editing, optimization)
4. Growth Strategy (SEO, engagement, promotion)
5. Monetization Roadmap
6. Analytics & Iteration

Be creative, data-driven, and practical.""",

    "fitness": """You are a certified Fitness & Diet Planning Assistant.

User Profile: {user_profile}
User Goal: {goal}

Create a comprehensive fitness and nutrition plan with:
1. Initial Assessment (fitness level, goals, timeline)
2. Workout Plan (weekly schedule, exercises, progression)
3. Nutrition Strategy (calories, macros, meal plan)
4. Tracking & Measurement
5. Lifestyle Integration
6. Safety & Considerations

Be motivating, realistic, and safety-conscious.""",

    "academic": """You are an Academic Research & Study Planner.

User Profile: {user_profile}
User Goal: {goal}

Create a comprehensive academic plan with:
1. Learning Analysis (subject breakdown, learning style)
2. Study Schedule (weekly calendar, daily sessions)
3. Learning Strategies (active recall, note-taking, practice)
4. Resource Recommendations
5. Assessment Preparation
6. Research Workflow (if applicable)

Be thorough, methodical, and evidence-based.""",

    "startup": """You are a Startup / Business Launch Consultant.

User Profile: {user_profile}
User Goal: {goal}

Create a comprehensive startup launch plan with:
1. Business Model Analysis
2. Market Validation
3. Product Development (MVP, timeline)
4. Go-to-Market Strategy
5. Operational Plan
6. Funding Strategy
7. 90-Day Launch Roadmap

Be practical, focused on validation, and milestone-driven."""
}


# ============================================================================
# GEMINI CLIENT
# ============================================================================

def call_gemini(prompt: str, temperature: float = 0.7) -> str:
    """Call Gemini API"""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"temperature": temperature}
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "AI Multi-Purpose Copilot API",
        "version": "1.0.0",
        "copilots": list(COPILOT_PROMPTS.keys()),
        "endpoints": {
            "generate_plan": "/api/generate",
            "get_plan": "/api/plans/{plan_id}",
            "export_plan": "/api/export",
            "create_session": "/api/sessions",
            "get_session": "/api/sessions/{session_id}",
            "history": "/api/history"
        },
        "docs": "/docs"
    }


@app.get("/api/copilots")
def list_copilots():
    """List available copilots"""
    copilots = {
        "career": {
            "name": "Career & Resume Mentor",
            "description": "Personalized career advice, resume reviews, and interview prep"
        },
        "youtube": {
            "name": "YouTube & Content Creation Coach",
            "description": "Launch and grow your YouTube channel or content brand"
        },
        "fitness": {
            "name": "Fitness & Diet Planning Assistant",
            "description": "Personalized workout plans and nutrition guidance"
        },
        "academic": {
            "name": "Academic Research & Study Planner",
            "description": "Master your studies with personalized learning plans"
        },
        "startup": {
            "name": "Startup / Business Launch Consultant",
            "description": "Turn your business idea into a launch-ready plan"
        }
    }
    return {"copilots": copilots}


@app.post("/api/sessions", response_model=SessionResponse)
def create_session():
    """Create a new user session"""
    session_id = SessionManager.create_session()
    session_data = SessionManager.get_session(session_id)
    return SessionResponse(**session_data)


@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    """Get session data"""
    session_data = SessionManager.get_session(session_id)
    return SessionResponse(**session_data)


@app.post("/api/generate", response_model=GeneratePlanResponse)
def generate_plan(
    request: GeneratePlanRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Generate a personalized plan using selected copilot"""
    
    # Validate copilot type
    if request.copilot_type not in COPILOT_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid copilot_type. Must be one of: {list(COPILOT_PROMPTS.keys())}"
        )
    
    # Get or create session
    if x_session_id:
        try:
            session_data = SessionManager.get_session(x_session_id)
        except HTTPException:
            x_session_id = SessionManager.create_session()
            session_data = SessionManager.get_session(x_session_id)
    else:
        x_session_id = SessionManager.create_session()
        session_data = SessionManager.get_session(x_session_id)
    
    # Update user profile if provided
    if request.user_profile:
        session_data["user_profile"] = request.user_profile.dict()
        SessionManager.update_session(x_session_id, session_data)
    
    # Prepare prompt
    user_profile_str = json.dumps(session_data["user_profile"], indent=2) if session_data["user_profile"] else "No profile set"
    
    prompt_template = COPILOT_PROMPTS[request.copilot_type]
    prompt = prompt_template.format(
        user_profile=user_profile_str,
        goal=request.goal
    )
    
    # Generate plan
    plan = call_gemini(prompt, request.temperature)
    
    # Create project
    plan_id = str(uuid.uuid4())
    project = {
        "plan_id": plan_id,
        "copilot_type": request.copilot_type,
        "goal": request.goal,
        "plan": plan,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to session
    SessionManager.add_project_to_session(x_session_id, project)
    
    # Save individual plan file
    plan_file = os.path.join(EXPORTS_DIR, f"{plan_id}.json")
    with open(plan_file, 'w') as f:
        json.dump(project, f, indent=2)
    
    return GeneratePlanResponse(
        **project,
        session_id=x_session_id
    )


@app.get("/api/plans/{plan_id}")
def get_plan(plan_id: str):
    """Get a specific plan by ID"""
    plan_file = os.path.join(EXPORTS_DIR, f"{plan_id}.json")
    
    if not os.path.exists(plan_file):
        raise HTTPException(status_code=404, detail="Plan not found")
    
    with open(plan_file, 'r') as f:
        plan_data = json.load(f)
    
    return plan_data


@app.post("/api/export")
def export_plan(request: ExportRequest):
    """Export a plan in specified format"""
    
    # Get plan
    plan_file = os.path.join(EXPORTS_DIR, f"{request.plan_id}.json")
    
    if not os.path.exists(plan_file):
        raise HTTPException(status_code=404, detail="Plan not found")
    
    with open(plan_file, 'r') as f:
        plan_data = json.load(f)
    
    title = request.title or f"{plan_data['copilot_type'].title()} Plan"
    
    if request.format == "markdown":
        content = f"""# {title}

*Generated: {plan_data['timestamp']}*

## Goal
{plan_data['goal']}

## Plan

{plan_data['plan']}

---
*Generated by AI Multi-Purpose Copilot*
"""
        filename = f"{request.plan_id}.md"
        filepath = os.path.join(EXPORTS_DIR, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return FileResponse(filepath, media_type="text/markdown", filename=filename)
    
    elif request.format == "json":
        return JSONResponse(content=plan_data)
    
    elif request.format == "text":
        content = f"{title}\n\n{plan_data['goal']}\n\n{plan_data['plan']}"
        filename = f"{request.plan_id}.txt"
        filepath = os.path.join(EXPORTS_DIR, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return FileResponse(filepath, media_type="text/plain", filename=filename)
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Must be one of: markdown, json, text"
        )


@app.get("/api/history", response_model=ProjectHistoryResponse)
def get_history(
    x_session_id: Optional[str] = Header(None),
    copilot_type: Optional[str] = None,
    limit: int = 10
):
    """Get project history"""
    
    if x_session_id:
        # Get session-specific history
        session_data = SessionManager.get_session(x_session_id)
        projects = session_data["projects"]
    else:
        # Get all projects
        projects = []
        for filename in os.listdir(EXPORTS_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(EXPORTS_DIR, filename), 'r') as f:
                    try:
                        project = json.load(f)
                        projects.append(project)
                    except:
                        continue
    
    # Filter by copilot type if specified
    if copilot_type:
        projects = [p for p in projects if p.get("copilot_type") == copilot_type]
    
    # Sort by timestamp and limit
    projects = sorted(projects, key=lambda x: x.get("timestamp", ""), reverse=True)
    projects = projects[:limit]
    
    return ProjectHistoryResponse(
        projects=projects,
        total=len(projects)
    )


@app.delete("/api/plans/{plan_id}")
def delete_plan(plan_id: str):
    """Delete a plan"""
    plan_file = os.path.join(EXPORTS_DIR, f"{plan_id}.json")
    
    if not os.path.exists(plan_file):
        raise HTTPException(status_code=404, detail="Plan not found")
    
    os.remove(plan_file)
    
    return {"message": "Plan deleted successfully", "plan_id": plan_id}


@app.get("/api/stats")
def get_stats():
    """Get API statistics"""
    
    # Count sessions
    sessions = len([f for f in os.listdir(SESSIONS_DIR) if f.endswith('.json')])
    
    # Count plans
    plans = len([f for f in os.listdir(EXPORTS_DIR) if f.endswith('.json')])
    
    # Count by copilot type
    copilot_counts = {}
    for filename in os.listdir(EXPORTS_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(EXPORTS_DIR, filename), 'r') as f:
                try:
                    project = json.load(f)
                    copilot_type = project.get("copilot_type", "unknown")
                    copilot_counts[copilot_type] = copilot_counts.get(copilot_type, 0) + 1
                except:
                    continue
    
    return {
        "total_sessions": sessions,
        "total_plans": plans,
        "plans_by_type": copilot_counts,
        "available_copilots": list(COPILOT_PROMPTS.keys())
    }


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
