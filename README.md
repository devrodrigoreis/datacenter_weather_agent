# Data Center Weather Agent

A LangGraph agent system that determines the weather forecast at a data center location using the Model Context Protocol (MCP).

**Architecture**: Custom StateGraph with explicit state management, modular nodes, and conditional routing following LangGraph 2026 best practices.

## Overview

This system answers the question: *"What is the weather forecast of the data center?"*

It autonomously:
0. Check the Intent Classification avoind answering questions that it is not supposed too answer or giving the same answer for any question
1. Discovers the data center's public IP address
2. Resolves the IP to geographic coordinates
3. Fetches the current weather forecast for that location
4. Synthesizes a natural language response

### Why This Architecture?

This implementation uses LangGraph's **custom StateGraph** approach rather than prebuilt agents because:

- **Full Transparency**: Every decision point is explicit and debuggable
- **Production-Ready**: Modular design supports testing, monitoring, and extension
- **Best Practices**: Follows LangGraph 2026 recommendations for stateful agents acquired in the Langgraph documentation: https://docs.langchain.com/oss/python/langgraph/workflows-agents
- **Maintainable**: Clear separation of concerns with typed state schemas

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Center Weather Agent                    │
│                                                                 │
│  ┌─────────────┐         ┌───────────────────────────────────┐  │
│  │   Agent     │  HTTP   │         MCP Server                │  │
│  │ (LangGraph) │◄───────►│  ┌────────┐ ┌─────────┐ ┌───────┐ │  │
│  │             │   SSE   │  │ ipify  │ │ip_to_geo│ │weather│ │  |
│  │  5 Nodes    │         │  └───┬────┘ └────┬────┘ └──┬────┘ │  │
│  │  3 Edges    │         │      │           │         │      │  │
│  │  9 State    │         │      ▼           ▼         ▼      │  │
│  │  Fields     │         │    ipify.org  ip-api.com  meteo   │  │
│  └─────────────┘         └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Components

#### 1. MCP Server (`server/`)
- **Technology**: Python, FastMCP, uvicorn
- **Port**: 8000 (SSE transport)
- **Tools**:
  - `ipify`: Get public IP address 
  - `ip_to_geo`: Convert IP to lat/lon (ip-api.com)
  - `weather_forecast`: Fetch weather data (Open-Meteo)
- **Features**: Input validation, error handling, structured logging

#### 2. LangGraph Agent (`agent/`)
- **Technology**: Python, LangGraph, LangChain, Google Gemini
- **Architecture**: Custom StateGraph
- **State**: 9 typed fields tracking workflow progress
- **Nodes**: 5 modular computation units
- **Edges**: 3 conditional routing functions
- **Features**: LLM fallback, detailed trace, error recovery

## State Management

All data flows through a strongly typed `AgentState`:

```python
class AgentState(TypedDict):
    question: str              # User's input query
    public_ip: str | None      # From ipify tool
    latitude: float | None     # From ip_to_geo tool  
    longitude: float | None    # From ip_to_geo tool
    weather_data: str | None   # From weather_forecast tool
    answer: str | None         # Final LLM response
    messages: list             # Conversation history
    error: str | None          # Error tracking
    current_step: str          # Progress indicator
```

## Graph Structure

```
START
  ↓
┌───────────────────────┐
│  intent classification│ → [custom logic]
└──────┬────────────────┘
       ↓
┌─────────────┐
│  get_ip     │ → [ipify tool]
└──────┬──────┘
       │ [conditional: success/error]
       ↓
┌──────────────────┐
│ resolve_location │ → [ip_to_geo tool]
└──────┬───────────┘
       │ [conditional: success/error]
       ↓
┌──────────────┐
│fetch_weather │ → [weather_forecast tool]
└──────┬───────┘
       │ [conditional: success/error]
       ↓
┌────────────────┐
│generate_answer │ → [LLM synthesis]
└───────┬────────┘
        ↓
       END

     [error] ← Any failure routes here
```

### Node Responsibilities

1. **get_ip_node**: Calls ipify tool, updates state with public_ip
2. **resolve_location_node**: Calls ip_to_geo, extracts lat/lon
3. **fetch_weather_node**: Calls weather_forecast with coordinates
4. **generate_answer_node**: LLM creates natural language response
5. **error_node**: Centralized error handling and user messaging

### Conditional Routing

Each edge validates the previous operation before proceeding:

```python
def route_after_ip(state: AgentState) -> Literal["resolve_location", "error"]:
    """Route to next node or error based on IP fetch result"""
    if state.get("error") or not state.get("public_ip"):
        return "error"
    return "resolve_location"
```

## Setup

### Prerequisites

- Python 3.10 or higher
- Fully tested on the version 3.14.2
- Google Gemini API key (free tier available)
- Optional: LongCat API key (fallback)

### Installation

1. **Clone or Download** this project

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**:
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```bash
   GOOGLE_API_KEY=your_gemini_api_key_here
   LONGCAT_API_KEY=your_longcat_api_key_here  # Optional
   ```
   
   Get API keys:
   - Gemini: https://ai.google.dev/
   - LongCat: https://longcat.chat/ (optional fallback)

## Usage

### Start the System

You need two terminal windows:

**Terminal 1 - Start MCP Server**:
```bash
python3 -m server.main
```

Expected output:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Run Agent**:
```bash
python3 -m agent.main
```

### Example Session

```
============================================================
Data Center Weather Agent (Custom StateGraph)
============================================================

Connected to MCP Server
LLM configured with fallback: Gemini -> LongCat

Graph structure built successfully

Agent Ready! (Type 'quit' to exit)
------------------------------------------------------------

Enter your question: What is the weather forecast of the data center?

============================================================
EXECUTION TRACE
============================================================

[Step 1: IP Discovery]
  Tool: ipify
  Result: 174.162.142.78

[Step 2: Location Resolution]
  Tool: ip_to_geo
  Input: 174.162.142.78
  Result: 40.3495, -111.8998

[Step 3: Weather Retrieval]
  Tool: weather_forecast
  Input: lat=40.3495, lon=-111.8998
  Result: Temperature: 0.2 C, Windspeed: 2.2 km/h

[Step 4: Answer Generation]
  Generated answer successfully

============================================================
FINAL ANSWER
============================================================

The data center, located at 40.3495°N, 111.8998°W, currently has a 
temperature of 0.2°C and a wind speed of 2.2 km/h.

Enter your question: Where is the data center located?

[... processes IP and location lookup ...]

FINAL ANSWER

The data center is located at coordinates 40.3495°N, 111.8998°W 
(approximately in Utah, United States).

Enter your question: quit
```

## File Structure

```
datacenter_weather_agent/
├── server/
│   ├── main.py              # MCP server with FastMCP
│   ├── tools.py             # Tool implementations
│   └── __init__.py
├── agent/
│   ├── main.py              # Custom StateGraph agent (~570 lines)
│   ├── client.py            # MCP client wrapper
│   └── __init__.py
├── README.md                # This file
├── ARCHITECTURE.md          # Deep technical documentation
├── .env.example             # API key template
├── .env                     # Your API keys (create this)
└── requirements.txt         # Python dependencies
```

## Key Features

### 1. Robust Input Validation

**Server-Side**:
- IP address format validation (IPv4/IPv6)
- IPv4 octet range checking (0-255)
- Coordinate range validation (lat: -90 to 90, lon: -180 to 180)
- Type checking on all parameters

**Client-Side**:
- Response structure validation
- Non-empty content verification
- Runtime error detection

### 2. Comprehensive Error Handling

**Per-Node Error Handling**:
```python
async def get_ip_node(state, tools):
    try:
        # ... tool execution ...
        return success_state
    except Exception as e:
        logger.error(f"Error in get_ip_node: {e}")
        return error_state
```

**Conditional Routing to Error Node**:
- Any node failure routes to centralized error handler
- User-friendly error messages
- Agent remains responsive (doesn't crash)

### 3. LLM Fallback Strategy ### -> Extra Feature <-

Automatic failover on rate limits:
```python
gemini_llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")
longcat_llm = ChatOpenAI(model="LongCat-Flash-Chat", ...)
llm = gemini_llm.with_fallbacks([longcat_llm])
```

### 4. Detailed Execution Trace

Every operation is logged:
- Tool calls with inputs and outputs
- State transitions
- Error conditions
- LLM reasoning

Good for debugging and monitoring.

### 5. Clean Execution Trace

The agent produces clean, focused output by:
- Suppressing verbose library logs (httpx, google_genai)
- Showing only essential step information
- Automatically handling LLM fallbacks silently

User-facing trace shows only what matters:
```
[Step 1: IP Discovery]
  Tool: ipify
  Result: 174.162.142.78
```

Backend logging still captures details for debugging.

## Tool Calling Strategy

The agent enforces a **strict sequential workflow**:

1. **IP Discovery** (ipify)
   - No prerequisites
   - Must succeed before proceeding

2. **Location Resolution** (ip_to_geo)
   - Requires: `public_ip` from step 1
   - Validates IP format before calling
   - Parses lat/lon from response

3. **Weather Retrieval** (weather_forecast)
   - Requires: `latitude` and `longitude` from step 2
   - Validates coordinate ranges
   - Returns formatted weather string

4. **Answer Generation** (LLM)
   - Requires: All collected data
   - Synthesizes natural language response
   - Maintains conversation context

**Validation**: Each conditional edge verifies prerequisites exist before routing to the next node.

## Production Enhancements

### Enable Checkpointing (Conversation Memory) 
### Its an extra intend for productions environments. 

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# In build_graph():
async with SqliteSaver.from_conn_string("./checkpoints.db") as memory:
    graph = workflow.compile(checkpointer=memory)

# In main():
config = {"configurable": {"thread_id": "user-123"}}
result = await graph.ainvoke(state, config=config)
```

### Add Human-in-the-Loop

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Compile with interrupt
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["generate_answer"]  # Pause before final answer
)

# Resume after approval
graph.invoke(None, config=config)  # Continue from interrupt
```

### Multi-Agent Extension

Create specialized sub-agents:

```python
weather_specialist = build_weather_subgraph()
location_specialist = build_location_subgraph()

workflow.add_node("weather", weather_specialist)
workflow.add_node("location", location_specialist)
workflow.add_conditional_edges("start", route_to_specialist, {...})
```

## Troubleshooting

### Server Connection Failed

**Symptom**: `connection refused` or `server not found`

**Solution**:
1. Verify server is running: `curl http://localhost:8000/sse`
2. Check no other process is using port 8000
3. Review server logs for startup errors

### LLM Rate Limit Errors

**Symptom**: `429 RESOURCE_EXHAUSTED`

**Solution**:
- Fallback to LongCat triggers automatically if configured
- Wait 60 seconds and retry
- Consider upgrading to Gemini paid tier

### Import Errors

**Symptom**: `ModuleNotFoundError`

**Solution**:
```bash
# Always run from project root:
cd /path/to/datacenter_weather_agent
python3 -m agent.main 
```

### Tool Validation Errors

**Symptom**: `Invalid IP address format` or `Invalid latitude`

**Solution**:
- Check server logs for detailed error
- Verify external APIs are accessible
- Test manually: `curl https://api.ipify.org?format=json`

## Performance

**Typical Metrics**:
- Cold start: 1-2 seconds (library loading)
- Tool execution: 3-5 seconds (3× HTTP requests)
- LLM synthesis: 0.5-1 second
- Total: ~5-8 seconds per query
- Memory: ~50MB runtime

**Bottlenecks**:
- External API latency (ipify, ip-api, open-meteo)
- LLM generation time - free tier api keys have a significant slower performance, please be patient
- Network conditions

## Limitations

1. **Sequential Execution**: Tools run in sequence (could be parallelized in some cases)
2. **Public IP Only**: Cannot determine location of internal/private IPs
3. **VPN Aware**: Location reflects VPN exit node if active
4. **No Retry Logic**: Failed tool calls immediately error (can add exponential backoff)
5. **Single Query**: Processes one question at a time (can add batch support)
6. **No Persistence**: State cleared between sessions (unless checkpointing enabled)

## Extending the Agent

### Add a New Tool

1. **Implement in `server/tools.py`**:
   ```python
   async def get_timezone(latitude: float, longitude: float) -> str:
       # Implementation
       return timezone_data
   ```

2. **Register in `server/main.py`**:
   ```python
   @mcp.tool()
   async def timezone(latitude: float, longitude: float) -> str:
       return await get_timezone(latitude, longitude)
   ```

3. **Add Node in `agent/main.py`**:
   ```python
   async def get_timezone_node(state, tools):
       tool = next(t for t in tools if t.name == "timezone")
       result = await tool.ainvoke({
           "latitude": state["latitude"],
           "longitude": state["longitude"]
       })
       return {**state, "timezone": result}
   
   workflow.add_node("get_timezone", get_timezone_node)
   workflow.add_edge("fetch_weather", "get_timezone")
   workflow.add_edge("get_timezone", "generate_answer")
   ```

### Modify Routing Logic

Change conditional edges to support alternative flows:

```python
def route_after_ip(state: AgentState):
    if state.get("error"):
        return "error"
    
    # New: Check if IP is internal/private
    ip = state.get("public_ip", "")
    if ip.startswith("192.168.") or ip.startswith("10."):
        return "handle_private_ip"  # New node
    
    return "resolve_location"
```

## Testing

### Unit Tests

Test individual nodes:

```python
import pytest
from agent.main import get_ip_node

@pytest.mark.asyncio
async def test_get_ip_node():
    mock_tools = [MockIPifyTool()]
    state = {"question": "test", "messages": []}
    
    result = await get_ip_node(state, mock_tools)
    
    assert result["public_ip"] is not None
    assert result["error"] is None
```

### Integration Tests

Test full graph execution:

```python
@pytest.mark.asyncio
async def test_full_workflow(mcp_server_running):
    graph = await build_graph(client, llm)
    initial_state = {...}
    
    final_state = await graph.ainvoke(initial_state)
    
    assert final_state["answer"] is not None
    assert "temperature" in final_state["answer"].lower()
```

### LangSmith Monitoring (From docs, last access Jan 30 2026)

Enable tracing for production debugging:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT="datacenter-weather"
```

Every execution will be traced in LangSmith dashboard.

## Security Considerations

1. **API Keys**: Never commit `.env` to version control
2. **Server Exposure**: MCP server has no authentication (local use only)
3. **Rate Limiting**: External APIs may ban abusive usage
4. **Input Validation**: Server validates all tool inputs
5. **Error Messages**: Avoid leaking sensitive data in logs

## Dependencies

- **httpx**: Async HTTP client
- **mcp**: Model Context Protocol SDK
- **langgraph**: Graph-based agent framework
- **langchain**: LLM abstraction layer
- **langchain-google-genai**: Gemini integration
- **langchain-openai**: OpenAI-compatible APIs (LongCat)
- **python-dotenv**: Environment variable management
- **uvicorn**: ASGI server

See `requirements.txt` for specific versions.

## Learn More

- [LangGraph Documentation](https://langchain.com/docs/langgraph)
- [MCP Specification](https://modelcontextprotocol.io/)
- [StateGraph Tutorial](https://langchain.com/docs/langgraph/tutorials/state_graph)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)

## License

This is a demonstration project for personal purposes. No warranty provided.

---

**Built with**: LangGraph Custom StateGraph (Production Best Practices, January 2026)
