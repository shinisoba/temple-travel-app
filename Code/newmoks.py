from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from groq import Groq
import os
import json
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0.6)

# GROQ client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# File paths
output_files = {
    "temple_list": "temple_list.json",
    "temples": "temples.json",
    "itinerary": "itinerary.md",
    "temple_history": "temple_history.md",
}

# Tool
search_tool = DuckDuckGoSearchRun()

# Load temple list and extract temple names using GROQ
try:
    with open(output_files['temple_list'], "r", encoding="utf-8") as f:
        temple_data = json.load(f)

    if isinstance(temple_data, list):
        temples = [temple["name"] for temple in temple_data if isinstance(temple, dict) and "name" in temple]
    elif isinstance(temple_data, dict) and "temples" in temple_data:
        temples = temple_data["temples"]
    else:
        raise ValueError("Unexpected structure in temple_list.json")

    if not temples:
        raise ValueError("No temples found in temple_list.json")

    with open(output_files['temples'], "w", encoding="utf-8") as f:
        json.dump({"temples": temples}, f, indent=2)

except Exception as e:
    raise RuntimeError(f"Failed to extract temples from JSON: {str(e)}")

# Shared Agent for both tasks
trip_agent = Agent(
    role="Temple Trip Planner",
    goal="Plan the perfect spiritual trip",
    backstory="Expert in ancient temples and travel logistics, with a deep understanding of cultural and spiritual routes.",
    tools=[search_tool],
    verbose=True,
    allow_delegation=True,
    llm=llm,
)

# Tasks
def create_itinerary_task(user_input, temples):
    temple_list = "\n".join(temples)
    return Task(
        description=(
            f"Generate a markdown-based travel itinerary for a temple tourism trip.\n\n"
            f"### Input from user:\n{user_input}\n\n"
            f"### Temples available:\n{temple_list}\n\n"
            f"Output should be well-structured markdown with time slots and temple names."
        ),
        expected_output="Markdown travel itinerary with temple names and timeslots",
        agent=trip_agent,
    )

def create_history_task(temples):
    temple_list = "\n".join(temples)
    return Task(
        description=(
            f"Write a markdown file with historical summaries of the following temples:\n{temple_list}\n\n"
            f"Each temple should have a header and a detailed yet concise paragraph."
        ),
        expected_output="Markdown file with temple names and historical descriptions.",
        agent=trip_agent,
    )

# Markdown cleaning
def clean_markdown(markdown_text):
    return re.sub(r"\n{3,}", "\n\n", markdown_text.strip())

# Request model
class UserInput(BaseModel):
    input_text: str

# Routes
@app.post("/generate_trip")
async def generate_trip(request: Request):
    try:
        body = await request.json()
        user_input = body.get("input_text", "")
        if not user_input:
            return JSONResponse(content={"error": "Missing input_text"}, status_code=400)

        itinerary_task = create_itinerary_task(user_input, temples)
        crew = Crew(
            agents=[trip_agent],
            tasks=[itinerary_task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()
        cleaned = clean_markdown(result)

        with open(output_files['itinerary'], "w", encoding="utf-8") as f:
            f.write(cleaned)

        return {"markdown": cleaned}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/generate_temple_history")
async def generate_history():
    try:
        history_task = create_history_task(temples)
        crew = Crew(
            agents=[trip_agent],
            tasks=[history_task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()
        cleaned = clean_markdown(result)

        with open(output_files['temple_history'], "w", encoding="utf-8") as f:
            f.write(cleaned)

        return {"markdown": cleaned}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/temples")
def get_temples():
    try:
        with open(output_files['temples'], "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"temples": data.get("temples", [])}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
