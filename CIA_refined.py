from dotenv import load_dotenv
load_dotenv()

import os
import requests

from rich import print

from tavily import TavilyClient

from langchain_groq import ChatGroq
from langchain.tools import tool

from langchain_core.messages import ToolMessage

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

#############################################################
# Configuration
#############################################################

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

tavily_client = TavilyClient()

#############################################################
# WEATHER TOOL
#############################################################

@tool
def get_weather(city: str) -> str:
    """
    Use ONLY when the user explicitly asks
    for the current weather of a city.

    Never use this tool for greetings,
    coding questions,
    explanations,
    or general conversation.
    """

    try:

        url = "https://api.openweathermap.org/data/2.5/weather"

        params = {
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        if str(data.get("cod")) != "200":
            return data.get("message", "Unable to fetch weather.")

        return f"""
Weather Report

City : {city}

Temperature : {data['main']['temp']} °C

Feels Like : {data['main']['feels_like']} °C

Humidity : {data['main']['humidity']} %

Condition : {data['weather'][0]['description']}
"""

    except Exception as e:
        return f"Weather Tool Error : {e}"


#############################################################
# NEWS TOOL
#############################################################

@tool
def get_news(query: str) -> str:
    """
    Use ONLY when the user explicitly asks
    for current news,
    latest updates,
    breaking news,
    or recent events.

    Never use for greetings.
    """

    try:

        response = tavily_client.search(
            query=query,
            topic="news",
            search_depth="basic",
            max_results=3
        )

        results = response.get("results", [])

        if not results:
            return "No news found."

        output = []

        for article in results:

            output.append(
f"""
Title:
{article["title"]}

Summary:
{article["content"]}

Source:
{article["url"]}
"""
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"News Tool Error : {e}"


#############################################################
# SYSTEM PROMPT
#############################################################

SYSTEM_PROMPT = """
You are an AI City Intelligence Assistant.

You have access to tools.

Use tools ONLY when necessary.

Use get_weather ONLY if the user explicitly asks
for the current weather of a city.

Never guess the city.

If the city is missing,
ask the user for the city name.

Use get_news ONLY when the user explicitly asks
for recent news or current events.

Never use tools for:

- greetings
- coding
- mathematics
- explanations
- general conversation

If tool output is sufficient,
answer naturally without calling tools again.
"""

#############################################################
# MODEL
#############################################################

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

#############################################################
# OPTIONAL MIDDLEWARE
#############################################################

@wrap_tool_call
def human_approval(request, handler):
    """
    Ask for approval before executing a tool.
    """

    tool_name = request.tool_call["name"]

    confirm = input(
        f"\nAgent wants to call '{tool_name}'. Approve? (y/n): "
    )

    if confirm.lower() != "y":

        return ToolMessage(
            content="Tool execution denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)

#############################################################
# CREATE AGENT
#############################################################

agent = create_agent(
    model=llm,
    tools=[get_weather, get_news],
    system_prompt=SYSTEM_PROMPT,
    middleware=[human_approval]      # remove this line if you don't want approval
)

#############################################################
# CHAT LOOP
#############################################################

print("=" * 70)
print("            CITY INTELLIGENCE AGENT")
print("=" * 70)
print("Type 'exit' to quit.")

while True:

    user_input = input("\nYou : ")

    if user_input.lower() == "exit":
        break

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
    )

    print("\nAssistant:\n")

    print(result["messages"][-1].content)