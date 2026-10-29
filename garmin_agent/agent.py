import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

# Load environment variables
load_dotenv()

# Import Garmin tools
from .tools import (
    garmin_login,
    get_user_summary,
    get_user_profile,
    get_steps_data,
    get_heart_rate_data,
    get_sleep_data,
    get_stress_data,
    get_body_battery,
    get_activities,
    get_activity_details,
    get_activities_by_date,
    get_devices,
    add_weigh_in,
    download_activity,
)

# Define the root agent for Garmin Fitness Assistant
root_agent = Agent(
    name="garmin_fitness_assistant",
    model="gemini-2.5-flash",  # Use Gemini 2.5 Flash as default
    description="An intelligent AI-powered fitness assistant that connects to your Garmin Connect data.",
    instruction="""You are an expert fitness AI assistant with deep knowledge of exercise physiology, 
    training methodologies, and health optimization. You have comprehensive access to Garmin Connect data 
    through your tools, including activity tracking, heart rate monitoring, sleep analysis, stress levels, 
    and body battery metrics.
    
    CRITICAL REQUIREMENTS:
    1. Authentication: You MUST call the garmin_login tool first to authenticate with Garmin Connect before calling any other tool.
    2. Data Gathering: Only after successful authentication, use appropriate tools to gather data.
    3. Analysis: Provide comprehensive analysis based on the user's question and available data.
    4. Insights: Offer actionable, personalized recommendations.
    5. Safety: Always prioritize user safety and recommend consulting healthcare providers when appropriate.
    
    Always maintain an encouraging, supportive, and motivating tone. If authentication fails, explain the issue clearly to the user.""",
    tools=[
        FunctionTool(garmin_login),
        FunctionTool(get_user_summary),
        FunctionTool(get_user_profile),
        FunctionTool(get_steps_data),
        FunctionTool(get_heart_rate_data),
        FunctionTool(get_sleep_data),
        FunctionTool(get_stress_data),
        FunctionTool(get_body_battery),
        FunctionTool(get_activities),
        FunctionTool(get_activity_details),
        FunctionTool(get_activities_by_date),
        FunctionTool(get_devices),
        FunctionTool(add_weigh_in),
        FunctionTool(download_activity),
    ]
)
