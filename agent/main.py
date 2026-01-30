"""
This is custom LangGraph StateGraph Implementation for Data Center Weather Agent

This module follows the current date best-practice LangGraph architecture using:
- Custom state management with Pydantic models
- Explicit node definitions for each operation phase  
- Conditional edges for dynamic routing
- Intent classification to validate question scope
- Comprehensive error handling and logging
- Human-readable execution trace

Architecture:
    User Input -> Intent Classification -> [Weather/Location Related?]
                         |                            |
                         NO -> Polite Refusal(extra)  YES -> IP Discovery -> Location Resolution 
                                                             |               |
                                                             [ipify]         [ip_to_geo]
                                                                   
                                                         -> Weather Fetch   ->      Answer Generation
                                                            |                       |
                                                            [weather_forecast]      [LLM]
"""

import asyncio
import os
import warnings
# Suppress warnings before importing packages that trigger them
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*create_react_agent has been moved.*")
warnings.filterwarnings("ignore", message=".*Core Pydantic V1 functionality.*")

from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
import logging
from pydantic import BaseModel, Field, SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from agent.client import MCPClient


# Load environment variables from .env file. 
load_dotenv()

# Configure logging - suppress verbose library output - the outputs where too verbose making the reading confuse. 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress long library loggers to keep trace clean
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('google_genai.models').setLevel(logging.WARNING)
logging.getLogger('google_genai._api_client').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('langchain').setLevel(logging.WARNING)


#Time to stardt defining the state schema for the agent graph
class AgentState(TypedDict):
    """
    The state schema for this agent graph.
    This defines all data that flows through the graph nodes.
    Using TypedDict allows LangGraph to validate and track state changes.
    """
    # User input
    question: str
    
    # Intent classification
    is_weather_question: bool | None
    
    # Tool results (accumulated through graph execution)
    public_ip: str | None
    latitude: float | None
    longitude: float | None
    weather_data: str | None
    
    # Final output
    answer: str | None
    
    # Conversation history for LLM context
    messages: Annotated[list[BaseMessage], "Conversation messages"]
    
    # Error tracking
    error: str | None
    
    # Progress tracking for trace output
    current_step: str

# Adding the NODES (Computation Units) - "best part"
#This node classifies the user intent to determine if the question is weather/location related.
# If not, the agent will refuse to answer.
# I think that it is crucial to prevent the agent from attempting to answer off-topic questions.
# whitout this step, the agent might try to provide weather/location info for unrelated queries, 
# or maybe be exploited to answer questions outside its scope.
# it was not of the requirements, but I think it is a good security practice to include it.

async def classify_intent_node(state: AgentState, llm) -> AgentState:
    """
    Node 0: Classify user intent to determine if question is weather/location related.
    
    This node uses the LLM to analyze the question and decide if it's asking about
    data center weather or location. If not, we should refuse the request.
    
    Args:
        state: Current agent state with user question
        llm: Language model instance
        
    Returns:
        Updated state with is_weather_question flag
    """
    print("\n[Step 0: Intent Classification]")
    
    try:
        # Construct classification prompt
        system_prompt = SystemMessage(content=(
            "You are a classifier for a data center weather agent. "
            "Analyze the user's question and determine if it's asking about:\n"
            "1. Weather at the data center (temperature, wind, forecast, etc.)\n"
            "2. Location of the data center (where, coordinates, IP address, etc.)\n\n"
            "Respond with ONLY 'YES' if the question is weather/location related, "
            "or 'NO' if it's about something else (time, date, general knowledge, etc.)."
        ))
        
        question_message = HumanMessage(content=f"Question: {state['question']}")
        
        # Invoke LLM for classification
        response = await llm.ainvoke([system_prompt, question_message])
        classification = response.content.strip().upper()
        
        is_weather_question = "YES" in classification
        
        print(f"  Question: {state['question'][:60]}..." if len(state['question']) > 60 else f"  Question: {state['question']}")
        print(f"  Classification: {'Weather/Location Related' if is_weather_question else 'Off-Topic'}")
        
        # Update state
        return {
            **state,
            "is_weather_question": is_weather_question,
            "current_step": "intent_classified",
            "messages": state["messages"] + [
                AIMessage(content=f"Intent classified: {classification}")
            ]
        }
    except Exception as e:
        logger.error(f"Error in classify_intent_node: {e}")
        # On classification error, assume it's weather-related to avoid false negatives
        return {
            **state,
            "is_weather_question": True,
            "current_step": "intent_classification_error",
            "error": None  # Don't fail the whole process
        }
# This node politely refuses to answer off-topic requests.
# If the question is out of scope, this node generates a polite refusal message    
async def refuse_request_node(state: AgentState) -> AgentState:
    """
    Node: Politely refuse off-topic requests.
    
    This node is reached when the question is not about weather or location.
    
    Args:
        state: Current agent state
        
    Returns:
        State with polite refusal message
    """
    print("\n[Request Refused: Off-Topic]")
    
    refusal_message = (
        "I'm sorry, but I'm specifically designed to answer questions about "
        "the data center's weather forecast and location. \n\n"
        "I can help you with:\n"
        "- 'What is the weather forecast of the data center?'\n"
        "- 'Where is the data center located?'\n"
        "- 'What are the data center's coordinates?'\n\n"
        f"Your question ('{state['question']}') appears to be about something else. "
        "Please ask about the data center's weather or location."
    )
    
    print(f"  Response: Providing scope clarification")
    
    return {
        **state,
        "answer": refusal_message,
        "current_step": "request_refused"
    }

# Now, I  will continue defining the rest of the nodes for IP discovery, 
#location resolution, weather fetching, and answer generation.

# Node 1: Fetch Public IP Address
# This node calls the 'ipify' tool through the MCP client to get the public IP address of the data center.
async def get_ip_node(state: AgentState, tools_client) -> AgentState:
    """
    Node 1: Fetch the public IP address of the data center.
    
    This node calls the 'ipify' tool through the MCP client.
    
    Args:
        state: Current agent state
        tools_client: MCP client with available tools
        
    Returns:
        Updated state with public_ip populated
    """
    print("\n[Step 1: IP Discovery]")
    
    try:
        # Find the ipify tool
        ipify_tool = next((t for t in tools_client if t.name == "ipify"), None)
        if not ipify_tool:
            raise RuntimeError("ipify tool not available")
        
        # Execute tool
        result = await ipify_tool.ainvoke({})
        public_ip = result.strip()
        
        print(f"  Tool: ipify")
        print(f"  Result: {public_ip}")
        
        # Update state
        return {
            **state,
            "public_ip": public_ip,
            "current_step": "ip_discovered",
            "messages": state["messages"] + [
                AIMessage(content=f"Discovered public IP: {public_ip}")
            ]
        }
    except Exception as e:
        logger.error(f"Error in get_ip_node: {e}")
        return {
            **state,
            "error": f"Failed to fetch IP: {str(e)}",
            "current_step": "error"
        }

# Node 2: Resolve Location from IP Address
# This node calls the 'ip_to_geo' tool to convert the discovered IP address into geographic coordinates.
async def resolve_location_node(state: AgentState, tools_client) -> AgentState:
    """
    Node 2: Resolve IP address to geographic coordinates.
    
    This node calls the 'ip_to_geo' tool using the IP from previous node.
    
    Args:
        state: Current agent state (must have public_ip)
        tools_client: MCP client with available tools
        
    Returns:
        Updated state with latitude and longitude populated
    """
    print("\n[Step 2: Location Resolution]")
    
    try:
        # Validate prerequisites
        if not state.get("public_ip"):
            raise RuntimeError("No public IP available in state")
        
        # Find the ip_to_geo tool
        geo_tool = next((t for t in tools_client if t.name == "ip_to_geo"), None)
        if not geo_tool:
            raise RuntimeError("ip_to_geo tool not available")
        
        # Execute tool
        result = await geo_tool.ainvoke({"ip": state["public_ip"]})
        
        # Parse coordinates from result
        # Result format: "latitude,longitude" or dict
        if isinstance(result, str):
            parts = result.strip().split(',')
            latitude = float(parts[0])
            longitude = float(parts[1])
        else:
            latitude = result.get("latitude")
            longitude = result.get("longitude")
        
        print(f"  Tool: ip_to_geo")
        print(f"  Input: {state['public_ip']}")
        print(f"  Result: {latitude}, {longitude}")
        
        # Update state
        return {
            **state,
            "latitude": latitude,
            "longitude": longitude,
            "current_step": "location_resolved",
            "messages": state["messages"] + [
                AIMessage(content=f"Located at coordinates: {latitude}, {longitude}")
            ]
        }
    except Exception as e:
        logger.error(f"Error in resolve_location_node: {e}")
        return {
            **state,
            "error": f"Failed to resolve location: {str(e)}",
            "current_step": "error"
        }

# Node 3: Fetch Weather Forecast
# This node calls the 'weather_forecast' tool to get the weather forecast for the resolved coordinates.
async def fetch_weather_node(state: AgentState, tools_client) -> AgentState:
    """
    Node 3: Fetch weather forecast for the coordinates.
    
    This node calls the 'weather_forecast' tool using coordinates from previous node.
    
    Args:
        state: Current agent state (must have latitude/longitude)
        tools_client: MCP client with available tools
        
    Returns:
        Updated state with weather_data populated
    """
    print("\n[Step 3: Weather Retrieval]")
    
    try:
        # Validate prerequisites
        if state.get("latitude") is None or state.get("longitude") is None:
            raise RuntimeError("No coordinates available in state")
        
        # Find the weather_forecast tool
        weather_tool = next((t for t in tools_client if t.name == "weather_forecast"), None)
        if not weather_tool:
            raise RuntimeError("weather_forecast tool not available")
        
        # Execute tool
        result = await weather_tool.ainvoke({
            "latitude": state["latitude"],
            "longitude": state["longitude"]
        })
        
        weather_data = result.strip()
        
        print(f"  Tool: weather_forecast")
        print(f"  Input: lat={state['latitude']}, lon={state['longitude']}")
        print(f"  Result: {weather_data}")
        
        # Update state
        return {
            **state,
            "weather_data": weather_data,
            "current_step": "weather_fetched",
            "messages": state["messages"] + [
                AIMessage(content=f"Weather data: {weather_data}")
            ]
        }
    except Exception as e:
        logger.error(f"Error in fetch_weather_node: {e}")
        return {
            **state,
            "error": f"Failed to fetch weather: {str(e)}",
            "current_step": "error"
        }
    
# Node 4: Generate Final Answer
# This node uses the LLM to synthesize all collected data into a human-readable answer.    
async def generate_answer_node(state: AgentState, llm) -> AgentState:
    """
    Node 4: Generate final natural language answer using LLM.
    
    This node uses the LLM to synthesize all collected data into a human-readable answer.
    
    Args:
        state: Current agent state (must have weather_data)
        llm: Language model instance
        
    Returns:
        Updated state with final answer
    """
    print("\n[Step 4: Answer Generation]")
    
    try:
        # Construct prompt for LLM
        system_prompt = SystemMessage(content=(
            "You are a helpful assistant. Based on the collected data, "
            "provide a concise answer to the user's question about the data center weather."
        ))
        
        context_message = HumanMessage(content=(
            f"User asked: {state['question']}\n\n"
            f"Data collected:\n"
            f"- Public IP: {state.get('public_ip', 'N/A')}\n"
            f"- Location: {state.get('latitude', 'N/A')}, {state.get('longitude', 'N/A')}\n"
            f"- Weather: {state.get('weather_data', 'N/A')}\n\n"
            f"Please provide a clear, concise answer."
        ))
        
        # Invoke LLM (will use fallback automatically if primary fails)
        try:
            response = await llm.ainvoke([system_prompt, context_message])
            answer = response.content
        except Exception as llm_err:
            # If both LLMs fail, provide helpful message
            if "quota" in str(llm_err).lower() or "rate" in str(llm_err).lower():
                print("  Note: LLM quota exceeded, using fallback...")
            raise
        
        print(f"  Generated answer successfully")
        
        # Update state
        return {
            **state,
            "answer": answer,
            "current_step": "complete",
            "messages": state["messages"] + [AIMessage(content=answer)]
        }
    except Exception as e:
        logger.error(f"Error in generate_answer_node: {e}")
        return {
            **state,
            "error": f"Failed to generate answer: {str(e)}",
            "current_step": "error"
        }
    
# Error Handling Node
# This node captures and logs errors from previous nodes, providing a user-friendly error message.
def error_node(state: AgentState) -> AgentState:
    """
    Error handling node.
    
    This node is reached when any previous node encounters an error.
    
    Args:
        state: Current agent state with error information
        
    Returns:
        State with error message as answer
    """
    print(f"\n[Error]: {state.get('error', 'Unknown error')}")
    
    return {
        **state,
        "answer": f"An error occurred: {state.get('error', 'Unknown error')}",
        "current_step": "error_handled"
    }


# Routing logic (Conditional Edges)
# Define the conditional routing between nodes based on state. 
# Conditional edge: Determine next node after intent classification.
def route_after_classification(state: AgentState) -> Literal["get_ip", "refuse_request"]:
    """
    Conditional edge: Determine next node after intent classification.
    
    Routes to 'get_ip' if question is weather/location related,
    otherwise routes to 'refuse_request' to politely decline.
    """
    if state.get("is_weather_question"):
        return "get_ip"
    return "refuse_request"

# Conditional edge: Determine next node after IP discovery.
def route_after_ip(state: AgentState) -> Literal["resolve_location", "error"]:
    """
    Conditional edge: Determine next node after IP discovery.
    
    Routes to 'resolve_location' if IP was successfully fetched,
    otherwise routes to 'error'.
    """
    if state.get("error"):
        return "error"
    if state.get("public_ip"):
        return "resolve_location"
    return "error"

# Conditional edge: Determine next node after location resolution.
def route_after_location(state: AgentState) -> Literal["fetch_weather", "error"]:
    """
    Conditional edge: Determine next node after location resolution.
    
    Routes to 'fetch_weather' if coordinates were successfully resolved,
    otherwise routes to 'error'.
    """
    if state.get("error"):
        return "error"
    if state.get("latitude") is not None and state.get("longitude") is not None:
        return "fetch_weather"
    return "error"

# Conditional edge: Determine next node after weather fetch.
def route_after_weather(state: AgentState) -> Literal["generate_answer", "error"]:
    """
    Conditional edge: Determine next node after weather fetch.
    
    Routes to 'generate_answer' if weather data was successfully fetched,
    otherwise routes to 'error'.
    """
    if state.get("error"):
        return "error"
    if state.get("weather_data"):
        return "generate_answer"
    return "error"



# Now, I will assemble the complete StateGraph using the defined nodes and routing logic.
# Graph design
# The graph starts with intent classification, then conditionally routes to either
# the IP discovery path or the refusal node. Each step includes error handling.
async def build_graph(mcp_client, llm):
    """
    Construct the StateGraph for the data center weather agent.
    
    Graph Structure:
        START -> classify_intent -> [conditional: weather? yes/no]
                      |
                      + -> YES -> get_ip -> resolve_location -> fetch_weather -> generate_answer -> END
                      |
                      + -> NO -> refuse_request -> END
              
        Any node can route to error node if failures occur.
    
    Args:
        mcp_client: MCP client instance for tool access
        llm: Language model instance
        
    Returns:
        Compiled graph ready for execution
    """
    # Get tools from MCP client
    tools = await mcp_client.get_tools()
    logger.info(f"Available tools: {[t.name for t in tools]}")
    
    # Create the StateGraph. 
    workflow = StateGraph(AgentState)
    
    # Create wrapper functions that properly await the async nodes. 
    async def classify_intent_wrapper(state: AgentState) -> AgentState:
        return await classify_intent_node(state, llm)
    
    async def get_ip_wrapper(state: AgentState) -> AgentState:
        return await get_ip_node(state, tools)
    
    async def resolve_location_wrapper(state: AgentState) -> AgentState:
        return await resolve_location_node(state, tools)
    
    async def fetch_weather_wrapper(state: AgentState) -> AgentState:
        return await fetch_weather_node(state, tools)
    
    async def generate_answer_wrapper(state: AgentState) -> AgentState:
        return await generate_answer_node(state, llm)
    
    async def refuse_request_wrapper(state: AgentState) -> AgentState:
        return await refuse_request_node(state)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent_wrapper)
    workflow.add_node("get_ip", get_ip_wrapper)
    workflow.add_node("resolve_location", resolve_location_wrapper)
    workflow.add_node("fetch_weather", fetch_weather_wrapper)
    workflow.add_node("generate_answer", generate_answer_wrapper)
    workflow.add_node("refuse_request", refuse_request_wrapper)
    workflow.add_node("error", error_node)
    
    # Set entry point - now starts with intent classification. 
    workflow.set_entry_point("classify_intent")
    
    # Add conditional edges with routing logic
    # First, check if question is weather-related
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "get_ip": "get_ip",
            "refuse_request": "refuse_request"
        }
    )
    
    workflow.add_conditional_edges(
        "get_ip",
        route_after_ip,
        {
            "resolve_location": "resolve_location",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "resolve_location",
        route_after_location,
        {
            "fetch_weather": "fetch_weather",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "fetch_weather",
        route_after_weather,
        {
            "generate_answer": "generate_answer",
            "error": "error"
        }
    )
    
    # Terminal nodes lead to END
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("refuse_request", END)
    workflow.add_edge("error", END)
    
    # Compile graph
    # Note: Checkpointing can be added for persistence/memory across sessions
    # For production, pass checkpointer=SqliteSaver.from_conn_string("./checkpoints.db")
    compiled_graph = workflow.compile()
    
    logger.info("Graph compiled successfully")
    return compiled_graph


#Main execution function 

async def main():
    """Main entry point for the custom graph implementation."""
    print("=" * 60)
    print("Data Center Weather Agent (Custom StateGraph)")
    print("=" * 60)
    
    # 1. Connect to MCP Server
    async with MCPClient(url="http://localhost:8000/sse") as client:
        print("\nConnected to MCP Server")
        
        # 2. Initialize LLMs (Primary: Gemini, Fallback: LongCat)
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("Warning: GOOGLE_API_KEY not found. Gemini will fail.")
        
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            google_api_key=google_api_key
        )
        
        longcat_api_key = os.getenv("LONGCAT_API_KEY")
        if not longcat_api_key:
            print("Warning: LONGCAT_API_KEY not found. Fallback not available.")
        
        longcat_llm = ChatOpenAI(
            model="LongCat-Flash-Chat",
            api_key=SecretStr(longcat_api_key) if longcat_api_key else None,
            base_url="https://api.longcat.chat/openai/v1"
        )
        
        # Configure fallback
        llm = gemini_llm.with_fallbacks([longcat_llm])
        print("LLM configured with fallback: Gemini -> LongCat")
        
        # 3. Build the graph
        graph = await build_graph(client, llm)
        print("\nGraph structure built successfully")
        
        # 4. Interactive loop
        print("\nAgent Ready! (Type 'quit' to exit)")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nEnter your question: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                
                if not user_input:
                    continue
                
                print("\n" + "=" * 60)
                print("EXECUTION TRACE")
                print("=" * 60)
                
                # Initialize state for this query
                initial_state = AgentState(
                    question=user_input,
                    is_weather_question=None,
                    public_ip=None,
                    latitude=None,
                    longitude=None,
                    weather_data=None,
                    answer=None,
                    messages=[HumanMessage(content=user_input)],
                    error=None,
                    current_step="started"
                )
                
                # Execute graph
                try:
                    final_state = await graph.ainvoke(initial_state)
                    
                    # Print final answer
                    print("\n" + "=" * 60)
                    print("FINAL ANSWER")
                    print("=" * 60)
                    print(f"\n{final_state.get('answer', 'No answer generated')}\n")
                    
                except Exception as graph_err:
                    print(f"\nError during graph execution: {graph_err}")
                    print("Please try again or rephrase your question.")
                
            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                print(f"\nCritical Error: {e}")


if __name__ == "__main__":
    import sys
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent stopped by user.")
    except Exception as e:
        print(f"\nFatal Error: {e}")