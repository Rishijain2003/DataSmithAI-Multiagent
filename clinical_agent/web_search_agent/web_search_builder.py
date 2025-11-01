import os
import json
import re
# Removed Groq import
from langgraph.graph import StateGraph, START, END
# FIX APPLIED HERE: Importing AIMessage from the correct core package
from langchain_core.messages import AIMessage, HumanMessage # Added HumanMessage for invoke calls
from typing import TypedDict, List, Dict, Union, Optional
from tavily import TavilyClient
from langgraph.graph import StateGraph
# Added ChatOpenAI import
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables (Essential for API Keys)
load_dotenv() 

# --- CONFIGURATION ---
# Changed API Key variable name and Model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") 
OPENAI_MODEL = "gpt-4o" # Using GPT-4o as a powerful default
SEARCH_QUERY_COUNT = 3

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set in environment.")
if TAVILY_API_KEY is None:
    raise ValueError("TAVILY_API_KEY is not set in environment.")




class SimpleWebState(TypedDict):
    """Represents the state of the simple web search agent."""
    user_question: str
    search_queries: List[str]
    web_results: List[Dict[str, Union[str, float]]]
    final_answer: AIMessage 

# ----------------------------------------------------
# 2. PROMPTS (Same)
# ----------------------------------------------------

QUERY_WRITER_PROMPT = """
You are an expert search query generator. Based on the user's question, generate exactly {number_queries} concise and distinct search queries that will maximize information retrieval for answering the question. 
Return only a JSON array of strings, like ["query 1", "query 2", ...].
User Question: {question}
"""

FINAL_ANSWER_PROMPT = """
You are an expert answer generator. Based on the original user question and the provided search results, synthesize a final, comprehensive answer. 

Use the search results only for factual information. Do not mention that you used a search tool.

Search Results:
{results}

User Question: {question}
"""



class WebSearchAgent:
    def __init__(self, openai_model: str = OPENAI_MODEL, query_count: int = SEARCH_QUERY_COUNT):
        
        self.openai_model = openai_model
        self.query_count = query_count

        # Initialize ChatOpenAI client
        self.openai_client = ChatOpenAI(
            model=self.openai_model,
            temperature=0.0,
            openai_api_key=OPENAI_API_KEY # Uses the environment variable directly
        )
        # Initialize Tavily client
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


    # Node 1: Generate Query (Updated for ChatOpenAI)
    def generate_query(self, state: SimpleWebState) -> SimpleWebState:
        """Generates search queries using the OpenAI LLM."""
        print("--- 1. Generating Search Queries ---")
        
        formatted_prompt = QUERY_WRITER_PROMPT.format(
            number_queries=self.query_count,
            question=state['user_question']
        )
        
        try:
            # Use ChatOpenAI's streaming/structure capability for JSON output
            structured_llm = self.openai_client.with_structured_output(
                schema={"type": "array", "items": {"type": "string"}}, 
                method="json_mode" # Enforce JSON output mode
            )
            
            # Invoke the structured LLM
            queries = structured_llm.invoke([HumanMessage(content=formatted_prompt)])

            if not isinstance(queries, list):
                # Fallback if the structured output parsing returns an unexpected format
                queries = [state['user_question']]
            
        except Exception as e:
            print(f"OpenAI API Error in query generation: {e}. Falling back to single query.")
            queries = [state['user_question']]
            
        print(f"Generated Queries: {queries}")
        return {"search_queries": queries}


    # Node 2: Web Search (Uses Tavily, client instantiation unchanged)
    def web_search(self, state: SimpleWebState) -> SimpleWebState:
        """Performs web research for each query using the Tavily Search API."""
        print("--- 2. Performing Web Research (via Tavily) ---")
        all_results = []
        
        for query in state['search_queries']:
            print(f"  Executing Tavily search for: {query}")
            try:
                search_response = self.tavily_client.search(
                    query=query, 
                    search_depth="advanced", 
                    max_results=3
                )
                
                for result in search_response['results']:
                    all_results.append({
                        "source": result.get('title', 'N/A'),
                        "snippet": result.get('content', 'No content available.'),
                        "url": result.get('url', 'N/A'),
                    })
            
            except Exception as e:
                print(f"Tavily Search Error for query '{query}': {e}")

        print(f"Aggregated {len(all_results)} search results.")
        return {"web_results": all_results}


    # Node 3: Finalize Answer (Updated for ChatOpenAI)
    def finalize_answer(self, state: SimpleWebState) -> SimpleWebState:
        """Synthesizes the final answer and appends the source URLs."""
        print("--- 3. Finalizing Answer ---")
        
        # 1. Format search results for LLM prompt
        results_str = "\n---\n".join([
            f"Source: {res.get('source', 'Unknown')} | URL: {res.get('url', 'N/A')}\nSnippet: {res['snippet']}"
            for res in state['web_results']
        ])
        
        formatted_prompt = FINAL_ANSWER_PROMPT.format(
            results=results_str,
            question=state['user_question']
        )

        try:
            # 2. Get LLM completion using ChatOpenAI.invoke
            completion = self.openai_client.invoke([HumanMessage(content=formatted_prompt)])
            final_text = completion.content
        except Exception as e:
            print(f"OpenAI API Error in final answer generation: {e}")
            final_text = f"Error generating final answer. Please check OpenAI API connection."

        # 3. Extract and format unique source URLs
        unique_urls = set()
        for res in state['web_results']:
            if 'url' in res and res['url'] != 'N/A':
                unique_urls.add(res['url'])

        citation_block = "\n\n***\n**Sources:**\n"
        citation_block += "\n".join([f"- {url}" for url in sorted(list(unique_urls))])
        
        # 4. Append citation block to the LLM's answer
        final_answer_with_citations = final_text + citation_block
        
        print("--- Final Answer Generated and Cited ---")
        # Return the output as an AIMessage object
        return {"final_answer": AIMessage(content=final_answer_with_citations)}


    # Build Graph
    def build(self):
        """Compiles the linear LangGraph workflow."""
        builder = StateGraph(SimpleWebState)

        # Define the nodes as methods of the class
        builder.add_node("generate_query", self.generate_query)
        builder.add_node("web_search", self.web_search)
        builder.add_node("finalize_answer", self.finalize_answer)

        # Define the linear workflow edges
        builder.add_edge(START, "generate_query")
        builder.add_edge("generate_query", "web_search")
        builder.add_edge("web_search", "finalize_answer")
        builder.add_edge("finalize_answer", END)

        return builder.compile()
    


# Example usage (Uncomment the final block in your main script to test):
agent = WebSearchAgent()
app = agent.build()
initial_state = {"user_question": "What's the latest research on SGLT2 inhibitors for kidney disease?"}
final_state = app.invoke(initial_state)
print(final_state['final_answer'].content)