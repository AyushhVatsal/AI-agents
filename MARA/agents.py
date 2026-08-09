from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from tools import web_search, scraper_url

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature = 0
)

#1st agent
def build_search_agent():
    return create_agent(
        model = llm,
        tools = [web_search],
        system_prompt=(
            "You are the Search Agent in a multi-agent research pipeline. "
            "Your task is to find high-quality sources for the Reader Agent. "
            "Use the web_search tool exactly once. "
            "After receiving the search results, do not call the tool again. "
            "Do not summarize or rewrite the search results. "
            "Return the search results with the title, complete URL, and snippet "
            "for every source. "
            "Preserve URLs exactly as provided by the tool. "
            "Do not invent, modify, shorten, or omit URLs."
        )
    )

#2nd agent
def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scraper_url],
        system_prompt=(
            "You are the Reader Agent in a multi-agent research pipeline. "

            "You receive search results containing multiple sources with titles, URLs, "
            "and snippets. "

            "Select the 2 most relevant and trustworthy sources. "
            "You MUST scrape exactly 2 different sources when at least 2 relevant URLs "
            "are available. "

            "Call scraper_url once for each selected URL. "
            "Do not scrape more than 2 URLs. "

            "After scraping the selected sources, synthesize the information. "

            "CRITICAL SOURCE-GROUNDING RULES: "
            "Use ONLY information contained in the provided search results and the "
            "content returned by scraper_url. "
            "Do not introduce information from your own knowledge. "
            "Do not mention sources that were not provided in the search results "
            "or scraped using scraper_url. "
            "Never invent a source, URL, organization, citation, statistic, or attribution. "

            "When reporting a finding, identify the exact source title or URL that "
            "provided the information. "
            
            "If a claim cannot be supported by the available sources, explicitly say "
            "that the available sources do not provide sufficient evidence."
        )
    )

#writer_chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer , write a clear, structured and insightful report."),
    ("human", """Write a detailed research report on the topic below,
    
    Topic: {topic}

    Research Gathered:
    {research}

    Structure the report as:

    1. Introduction
    2. Background
    3. Sources
    4. Results
    5. Conclusion

    Be deatailed, factual and professional."""
    )
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a critical review writer , write a clear, structured and insightful report."),
    ("human", """Write a detailed review report on the topic below,

    Report: {report}

    Respond in this format:

    Score: X/10

    Strength:

    Areas to Improve:

    One line verdict:

    """)
])

critic_chain = critic_prompt | llm | StrOutputParser()