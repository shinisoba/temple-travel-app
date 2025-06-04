import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from crewai import Agent, Crew, Task, Process, LLM
import os
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
import json
import re
from time import sleep
from pathlib import Path

import ssl
from urllib3 import poolmanager

BASE_DIR = Path(__file__).parent

# ----------- FastAPI Setup -------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputQuery(BaseModel):
    city: str
    start_date: str
    end_date: str
    no_of_days: str
    language: str

# ----------- CrewAI Logic -------------
os.environ['SERPER_API_KEY'] = "insert-api-key"

llm = LLM(
    model="gemini/gemini-1.5-flash",
    temperature=0.7,
    api_key="insert-api-key",
    request_timeout=120,
    max_retries=2,
    delay_between_retries=20,
    ssl_verify=False
)

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

def extract_temples_from_md(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        temples_section = re.search(r'## Temples to Visit\n(.*?)(\n|$)', content, re.DOTALL)
        if temples_section:
            temple_items = re.findall(r'-\s*(.*?)\n', temples_section.group(1))
            return temple_items
        return []
    except Exception as e:
        print(f"Error parsing markdown: {e}")
        return []

# Agents
temple_curator = Agent(
    role="Temple Expert",
    goal="Identify important temples in specified city",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory="Religious tourism specialist with deep knowledge of temple locations",
    llm=llm,
)

temple_guide = Agent(
    role="Historical Researcher",
    goal="Provide accurate temple histories from the list in requested language",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=("Multilingual historian specializing in religious architecture and its significance"
               "With immense knowledge in the history of temples, the city"
               "and the religion the temple belongs to"
               "you excel at finding accurate, engaging, informative content"
               "that covers the origins, significane, cultural insights, and spiritual importance of each temple."),
    llm=llm,
)

custom_guide = Agent(
    role="Travel Coordinator",
    goal="Create practical itineraries with logistics",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=("You are an experienced travel planner with a focus on religious tours"
               "With deep and updated knowledge of temple opening and closing times,"
               "city traffic, locations in the city, travel by different modes of transportation, and feasibility"
               "you excel at planning the logistics of the trip, showing the most efficient routes to cover as much as possible within the input duration"
               "based on the customer's mode of transportation, and presenting all the information in a sequential way thats easy to grasp"
               "Make sure to include the best visiting hours to avoid crowds."),
    llm=llm,
)

# Tasks
temple_list_task = Task(
    description="List significant temples in {city} visitable in {no_of_days} days",
    expected_output="JSON list of temple names with locations, with no comments or anything unnecessary in the json file.",
    agent=temple_curator,
    output_file="temple_list.json"
)

custom_trip_task = Task(
    description="Create {language} itinerary for {city} from {start_date} to {end_date}. "
                "Include 3 hotels (Budget/Mid/Luxury), transport costs, and temple schedule. "
                "Markdown format with clear '## Temples to Visit' section listing temples as bullet points.",
    expected_output="Detailed markdown itinerary with explicit temple list section",
    agent=custom_guide,
    input_from=temple_list_task,
    output_file="travel_itinerary.md"
)

info_task = Task(
    description="Generate accurate historical info in {language} for these temples: {temples}."
                "Focus on architectural style, founding period, and religious significance."
                "Verify facts against reputable sources.",
    expected_output="Markdown document with 150-200 word entries per temple",
    agent=temple_guide,
    output_file="temple_history.md"
)

# Crews
planning_crew = Crew(
    agents=[temple_curator, custom_guide],
    tasks=[temple_list_task, custom_trip_task],
    verbose=True,
)

history_crew = Crew(
    agents=[temple_guide],
    tasks=[info_task],
    verbose=True,
)

# Run full pipeline

@app.post("/generate_trip")
async def generate(query: InputQuery):
    output_files = {
        'temple_list': BASE_DIR / "temple_list.json",
        'temples': BASE_DIR / "temples.json",
        'itinerary': BASE_DIR / "travel_itinerary.md",
        'history': BASE_DIR / "temple_history.md"
    }
    
    try:
        # Existing pipeline logic
        trip_details = query.dict()
        
        # Generate Itinerary First
        try:
            planning_crew.kickoff(inputs=trip_details)
        except Exception as e:
            raise RuntimeError(f"Planning crew failed: {str(e)}")

        """temple_data = None
        raw_content = ""
        for attempt in range(5):
            try:
                sleep(30 + (attempt * 10))
                with open(output_files['temple_list'], "r", encoding="utf-8") as f:
                    raw_content = f.read().strip()
                    
                # Remove JSON code blocks and sanitize
                sanitized = re.sub(r'^```json|```$', '', raw_content, flags=re.MULTILINE)
                sanitized = re.sub(r'<[^>]+>', '', sanitized)
                
                temple_data = json.loads(sanitized)
                break
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Temple list read attempt {attempt+1} failed: {e}")
                if attempt == 4:
                    raise ValueError(f"Invalid temple list format after 5 attempts: {raw_content[:200]}...")
                continue"""
        """if not isinstance(temple_data, list):
            raise ValueError("Temple list is not a JSON array")
            
        temples = [entry["name"] for entry in temple_data if isinstance(entry, dict) and "name" in entry]
        if not temples:
            raise ValueError("No valid temples found in temple list")"""
        from groq import Groq

        client = Groq(api_key="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")
        try:
            with open("temple_list.json", "r", encoding="utf-8") as f:
                temple_data = json.load(f)
            
            # Extract just the temple names from the JSON structure
            temples = [temple["name"] for temple in temple_data if "name" in temple]
            
            if not temples:
                raise ValueError("No temples found in temple_list.json")

        except Exception as e:
            print(f"Error reading temple list: {e}")
            temples = []

        # Validate temple list before proceeding
        if not temples:
            raise ValueError("No temples found! Check temple_list.json format")

        # Save validated list
        with open("temples.json", "w", encoding="utf-8") as f:
            json.dump({"temples": temples}, f, indent=2)


        # Get Itinerary Immediately
        with open(output_files['itinerary'], "r", encoding="utf-8") as f:
            itinerary = f.read()

        # Generate History Separately
        history = ""
        try:
            history_crew.kickoff(inputs={'language': trip_details['language'], 'temples': temples})
            with open(output_files['history'], "r", encoding="utf-8") as f:
                history = f.read()
        except Exception as e:
            history = f"History generation failed: {str(e)}"

        return {
            "itinerary": itinerary,
            "history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# --------- Start FastAPI server --------
if __name__ == "__main__":
    uvicorn.run("moksha:app", host="0.0.0.0", port=8000, reload=True)
