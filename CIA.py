from dotenv import load_dotenv
load_dotenv()

import os
import requests

from rich import print

from tavily import TavilyClient

from langchain_groq import ChatGroq
from langchain.tools import tool

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage
)

#############################################################
#* Configuration
#############################################################

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

tavily_client = TavilyClient()

#############################################################
#! WEATHER TOOL
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

        response = requests.get(url, params=params)

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
#! NEWS TOOL
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

        if len(results) == 0:
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
#* SYSTEM PROMPT
#############################################################

SYSTEM_PROMPT = """
You are an AI City Intelligence Assistant.

You have access to tools.

Use tools ONLY when necessary.

Use get_weather ONLY if the user asks
for the current weather.

Use get_news ONLY if the user asks
for current or breaking news.

Never use tools for:

- greetings
- programming
- coding
- mathematics
- explanations
- general knowledge

If a tool has already been used and
you have enough information,
produce the final answer.

Do not repeatedly call tools.
"""

#############################################################
#* MODEL
#############################################################

model = ChatGroq(

    model="llama-3.3-70b-versatile",

    temperature=0

)

#############################################################
#* TOOL REGISTRY
#############################################################

tool_map = {

    "get_weather": get_weather,

    "get_news": get_news

}

#############################################################
#* TOOL BINDING
#############################################################

model_with_tools = model.bind_tools(

    [

        get_weather,

        get_news

    ]

)

#############################################################
#* MEMORY
#############################################################

messages = [

    SystemMessage(

        content=SYSTEM_PROMPT

    )

]

#############################################################
#* START AGENT
#############################################################

print("=" * 70)
print("            CITY INTELLIGENCE AGENT")
print("=" * 70)
print("Type 'exit' to quit.")

#############################################################
#* MAIN LOOP
#############################################################

while True:

    user_input = input("\nYou : ")

    if user_input.lower() == "exit":
        break

    
    #* ADD HUMAN MESSAGE
    

    messages.append(
        HumanMessage(content=user_input)
    )

    
    #* FIRST LLM CALL
    

    ai_message = model_with_tools.invoke(messages)

    messages.append(ai_message)

    
    #* NO TOOL REQUIRED
    

    if not ai_message.tool_calls:

        print("\nAssistant:\n")

        print(ai_message.content)

        continue

    
    #* EXECUTE ALL TOOLS
    

    for tool_call in ai_message.tool_calls:

        tool_name = tool_call["name"]

        tool = tool_map[tool_name]

        print(f"\nUsing Tool : {tool_name}")

        tool_result = tool.invoke(tool_call["args"])

        print(tool_result)

        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"]
        )

        messages.append(tool_message)
    
    #* FINAL LLM CALL
    

    final_response = model_with_tools.invoke(messages)

    messages.append(final_response)

    
    #* PRINT FINAL ANSWER
    

    print("\nAssistant:\n")

    print(final_response.content)