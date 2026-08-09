from arrow import get
from dotenv import load_dotenv
load_dotenv()
import os
import requests

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print
API_KEY = os.getenv("OPENWEATHER_API_KEY")
#now lets create some tools

tavily = TavilyClient()
#weather tool
@tool
def get_weather(city:str) -> str:
    """
    Get the current weather for a city.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    data = response.json()

    print("DEBUG", data)

    if str(data.get("code")) != 200:
        return f"Error: {data.get('message','Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"
# tavily news too
@tool
def get_news(query: str) -> str:
    """
    Search the web for the latest news on a topic.
    Use this tool for recent events, breaking news,
    or current affairs.
    """
    try:
        response = tavily.search(
            query=query,
            topic="news",
            search_depth="basic",
            max_results=3
        )

        results = response.get("results", [])

        if not results:
            return f"No news found for '{query}'."

        news = []

        for article in results:
            news.append(
                f"""
Title: {article['title']}

Summary:
{article['content']}

Source:
{article['url']}
"""
            )

        return "\n\n".join(news)

    except Exception as e:
        return f"Error fetching news: {e}"

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

model_tools = model.bind_tools([
    get_weather,
    get_news
])
# Agent loop

messages = []

print("City Intelligence System")
print("type exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == 'exit':
        break
    messages.append(HumanMessage(content = user_input))

    while True:
        result = model_tools.invoke(messages)

        messages.append(result)

        # if tool is required
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']

                # Human in the loop
                confirm = input(f"Agent wants to call {tool_name} Approve(y/n) : ")

                if confirm.lower() == 'n':
                    print("Tool call denied and i can not get the latest information.")
                    break

                # execute tool
                tool_result = tools[tool_name].invoke(tool_call)
                messages.append(ToolMessage(
                    content = tool_result,
                    tool_call_id = tool_call['id']
                ))

            continue

        else:
            print(result.content)    