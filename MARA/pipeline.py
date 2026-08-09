from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain
from langchain_core.messages import ToolMessage

def run_research_pipeline(topic: str) -> dict:

    state = {}

    #search agent working
    print("\n" + " ="*50)
    print("Step 1: Search Agent...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"Find recent, reliable and detailed information about: {topic}"
            )
        ]
    })

    tool_messages = [
        message
        for message in search_result["messages"]
        if isinstance(message, ToolMessage)
    ]

    state["search_results"] = tool_messages[-1].content

    print("Search result: ", state["search_results"])

    #reader agent working
    print("\n" + " ="*50)
    print("Step 2: Reader Agent...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages" : [("user",
            f"Based on the following search results about '{topic}',"
            f"pick the most relevant URL and scrape it for deeper content. \n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })
    state["scraped_content"] = reader_result['messages'][-1].content

    print("Scraped content: ", state["scraped_content"])

    #writer_chain

    print("\n" + " ="*50)
    print("Step 3: Writer is drafting the report...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS: \n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT: \n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic":topic,
        "research": research_combined
    })

    print("\n Final report \n", state['report'])

    #critic report

    print("\n" + " ="*50)
    print("Step 4: Critic is reviewing the report...")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report": state['report']
    })

    print("\n Final critic report \n", state['feedback'])

    return state

if __name__ == '__main__':
    topic = input("\nEnter the research topic: ")
    run_research_pipeline(topic)