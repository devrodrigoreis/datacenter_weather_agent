# Architecture
## System Design Philosophy

This project demonstrates **the LangGraph architecture** following 2026 best practices for building stateful, maintainable AI agents.

### Design Principles

1. **Explicit Over Implicit**: All state, nodes, and transitions are explicitly defined
2. **Single Responsibility**: Each node performs one logical operation
3. **Fail-Safe**: Every node includes error handling with graceful degradation
4. **Observable**: Comprehensive logging and tracing at every step
5. **Extensible**: Clear patterns for adding tools, nodes, or agents

---

## Component Architecture

### 1. MCP Server Layer

**File**: `server/main.py`

**Responsibility**: Tool discovery and execution gateway

**Implementation**:
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DataCenter Weather Tools")

@mcp.tool()
async def ipify() -> str:
    """Get public IP address"""
    return await get_public_ip()
```

**Transport**: Server-Sent Events (SSE) over HTTP
- **Why SSE**: Efficient for streaming tool results
- **Port**: 8000
- **Endpoint**: `/sse`

**Tool Registration**:
- Tools auto-discovered via `@mcp.tool()` decorator
- Schemas generated from function signatures
- OpenAPI-compatible descriptions

---

### 2. MCP Client Layer

**File**: `agent/client.py`

**Responsibility**: Bridge between MCP server and LangChain tools

**Key Methods**:

```python
class MCPClient:
    async def get_tools(self) -> list[StructuredTool]:
        """
        Fetch tools from MCP server and convert to LangChain format.
        
        Flow:
        1. HTTP GET to /sse (establish SSE connection)
        2. Send list_tools message
        3. Parse MCP tool definitions  
        4. Dynamically create Pydantic models for args
        5. Wrap in LangChain StructuredTool
        """
```

**Dynamic Schema Generation**:

```python
def _create_langchain_tool(self, mcp_tool):
    # Extract schema from MCP tool
    fields = {}
    for name, schema in mcp_tool.inputSchema["properties"].items():
        python_type = self._map_json_type(schema["type"])
        fields[name] = (python_type, Field(...))
    
    # Create Pydantic model dynamically
    ArgsModel = create_model(f"{mcp_tool.name}Arguments", **fields)
    
    # Wrap in async callable
    async def _tool_wrapper(**kwargs):
        result = await self.session.call_tool(mcp_tool.name, arguments=kwargs)
        return "\n".join([c.text for c in result.content if c.type == "text"])
    
    return StructuredTool.from_function(
        coroutine=_tool_wrapper,
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=ArgsModel
    )
```

**Why This Matters**:
- LLM receives properly formatted tool schemas
- Type validation happens automatically
- No manual tool definition needed
- This is a must when dealing with Langgraph to avoid errors

---

### 3. Agent Layer: StateGraph Architecture

**File**: `agent/main.py` (~570 lines)

#### State Management

**State Schema**:

```python
class AgentState(TypedDict):
    # User interaction
    question: str
    
    # Tool results (pipeline data)
    public_ip: str | None
    latitude: float | None
    longitude: float | None
    weather_data: str | None
    
    # Output
    answer: str | None
    
    # Internal management
    messages: Annotated[list[BaseMessage], "Conversation history"]
    error: str | None
    current_step: str  # For trace output
```

**Why TypedDict**:
- Type checking at development time
- Clear contract between nodes
- Runtime validation via LangGraph
- IDE autocomplete support



#### Node Implementation Pattern

All nodes follow this pattern:

```python
async def node_name(state: AgentState, dependencies) -> AgentState:
    """
    Single-responsibility node.
    
    Args:
        state: Current state (immutable)
        dependencies: External resources (tools, LLM)
        
    Returns:
        New state dict (state update)
    """
    logger.info("NODE: node_name - Starting operation")
    print(f"\n[Step X: Operation Name]")
    
    try:
        # 1. Validate prerequisites
        if not state.get("required_field"):
            raise RuntimeError("Missing required_field")
        
        # 2. Execute operation
        result = await perform_operation(...)
        
        # 3. Log success
        print(f"  Result: {result}")
        
        # 4. Return state update
        return {
            **state,  # Spread existing state
            "new_field": result,
            "current_step": "step_completed",
            "messages": state["messages"] + [AIMessage(content=f"...")]
        }
        
    except Exception as e:
        # 5. Error handling
        logger.error(f"Error in node_name: {e}")
        return {
            **state,
            "error": str(e),
            "current_step": "error"
        }
```
- This pattern facilitates future maintenances and understanding


**State Update Strategy**:
- Nodes return partial state updates (dicts)
- LangGraph merges updates into current state
- Previous fields persist unless overwritten
- Immutable update pattern (functional)

#### Conditional Routing

**Routing Functions** act as guards:

```python
def route_after_ip(state: AgentState) -> Literal["resolve_location", "error"]:
    """
    Decide next node based on current state.
    
    Decision logic:
    - If error exists → error node
    - If public_ip missing → error node  
    - Otherwise → continue to next step
    """
    if state.get("error"):
        return "error"
    
    if not state.get("public_ip"):
        return "error"
    
    return "resolve_location"
```

**Why Conditional Edges**:
- Explicit validation before each step
- Early error detection
- Self-documenting workflow logic
- Enables complex branching

#### Graph Construction

```python
async def build_graph(mcp_client, llm):
    # 1. Create graph
    workflow = StateGraph(AgentState)
    
    # 2. Add all nodes
    workflow.add_node("get_ip", get_ip_wrapper)
    workflow.add_node("resolve_location", resolve_location_wrapper)
    # ... more nodes
    
    # 3. Set entry point
    workflow.set_entry_point("get_ip")
    
    # 4. Add conditional edges
    workflow.add_conditional_edges(
        "get_ip",           # Source node
        route_after_ip,     # Routing function
        {
            "resolve_location": "resolve_location",  # Map return value → target node
            "error": "error"
        }
    )
    
    # 5. Add terminal edges
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("error", END)
    
    # 6. Compile
    return workflow.compile()
```

---

## Data Flow

### Complete Execution Flow

```
User Input: "What is the weather forecast of the data center?"
    ↓
Initialize State:
    {
        question: "What is the weather...",
        public_ip: None,
        latitude: None,
        longitude: None,
        weather_data: None,
        answer: None,
        messages: [HumanMessage(...)],
        error: None,
        current_step: "started"
    }
    ↓
┌────────────────────────────────────────────┐
│ Node: get_ip                               │
│ - Calls ipify tool                         │
│ - Result: "174.162.142.74"                 │
│ - Updates: state.public_ip = "174...."    │
│ - Updates: state.current_step = "ip_..."  │
└────────────────────┬───────────────────────┘
                     ↓
      Route: route_after_ip(state)
      - Check: state.error? No
      - Check: state.public_ip? Yes
      - Decision: "resolve_location"
                     ↓
┌────────────────────────────────────────────┐
│ Node: resolve_location                     │
│ - Calls ip_to_geo tool                     │
│ - Input: state.public_ip                   │
│ - Result: "40.3495,-111.8998"              │
│ - Parses: lat=40.3495, lon=-111.8998       │
│ - Updates: state.latitude, state.longitude │
└────────────────────┬───────────────────────┘
                     ↓
      Route: route_after_location(state)
      - Check: state.error? No
      - Check: state.latitude & longitude? Yes
      - Decision: "fetch_weather"
                     ↓
┌────────────────────────────────────────────┐
│ Node: fetch_weather                        │
│ - Calls weather_forecast tool              │
│ - Input: state.latitude, state.longitude   │
│ - Result: "Temperature: 0.2 C, ..."        │
│ - Updates: state.weather_data = "Temp..." │
└────────────────────┬───────────────────────┘
                     ↓
      Route: route_after_weather(state)
      - Check: state.error? No
      - Check: state.weather_data? Yes
      - Decision: "generate_answer"
                     ↓
┌────────────────────────────────────────────┐
│ Node: generate_answer                      │
│ - Calls LLM with context:                  │
│   * User question                          │
│   * Collected data (IP, coords, weather)   │
│ - LLM synthesizes response                 │
│ - Updates: state.answer = "The data ..."   │
└────────────────────┬───────────────────────┘
                     ↓
      Edge: generate_answer → END
                     ↓
Final State:
    {
        question: "What is the weather...",
        public_ip: "174.162.142.74",
        latitude: 40.3495,
        longitude: -111.8998,
        weather_data: "Temperature: 0.2 C, Windspeed: 2.2 km/h",
        answer: "The data center, located at 40.3495°N, 111.8998°W, currently has...",
        messages: [HumanMessage(...), AIMessage(...), ...],
        error: None,
        current_step: "complete"
    }
```

---

## Error Handling Strategy

### Multi-Layer Defense

1. **Server Layer** (`server/tools.py`):
   ```python
   async def get_location_from_ip(ip_address: str):
       # Input validation
       if not validate_ip_format(ip_address):
           raise ValueError("Invalid IP address format")
       
       try:
           # External API call
           response = await client.get(url)
           response.raise_for_status()
           
           # Response validation
           if data.get("status") == "fail":
               raise ValueError("Could not resolve location")
           
           return {"latitude": lat, "longitude": lon}
           
       except Exception as e:
           logger.error(f"Error: {e}")
           raise RuntimeError(f"Failed to fetch location: {e}")
   ```

2. **Client Layer** (`agent/client.py`):
   ```python
   async def _tool_wrapper(**kwargs):
       result = await self.session.call_tool(name, arguments=kwargs)
       
       # Validate response structure
       if not result or not hasattr(result, 'content'):
           raise RuntimeError("Tool returned invalid response")
       
       # Validate content
       text_content = [c.text for c in result.content if c.type == "text"]
       if not text_content:
           raise RuntimeError("Tool returned no text content")
       
       return "\n".join(text_content)
   ```

3. **Node Layer** (`agent/main.py`):
   ```python
   async def get_ip_node(state, tools):
       try:
           # Business logic
           result = await tool.ainvoke({})
           return success_state
       except Exception as e:
           # Capture error in state
           logger.error(f"Error in get_ip_node: {e}")
           return {**state, "error": str(e), "current_step": "error"}
   ```

4. **Graph Layer** (conditional routing):
   ```python
   def route_after_ip(state):
       # Check for errors before proceeding
       if state.get("error"):
           return "error"  # Route to error handler
       # ... validation logic
   ```

5. **Application Layer** (`agent/main.py` main loop):
   ```python
   try:
       final_state = await graph.ainvoke(initial_state)
       print(final_state["answer"])
   except Exception as graph_err:
       print(f"Error during graph execution: {graph_err}")
       print("Please try again or rephrase your question.")
   ```

### Error Recovery

**Error Node** provides graceful degradation:

```python
def error_node(state: AgentState) -> AgentState:
    """
    Centralized error handler.
    Ensures user gets helpful message instead of crash.
    """
    error_msg = state.get("error", "Unknown error occurred")
    logger.info(f"NODE: error_node - Handling error: {error_msg}")
    print(f"\n[Error]: {error_msg}")
    
    return {
        **state,
        "answer": f"Sorry, an error occurred: {error_msg}. Please try again.",
        "current_step": "error_handled"
    }
```

---

## LLM Integration

### Fallback Strategy

```python
# Primary LLM
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Fallback LLM
longcat_llm = ChatOpenAI(
    model="LongCat-Flash-Chat",
    api_key=os.getenv("LONGCAT_API_KEY"),
    base_url="https://api.longcat.chat/openai/v1"
)

# Automatic failover
llm = gemini_llm.with_fallbacks([longcat_llm])
```

**Fallback Triggers**:
- 429 Rate Limit Exceeded
- 503 Service Unavailable
- Network timeouts
- Any API error from primary LLM

### Answer Generation Logic

```python
async def generate_answer_node(state: AgentState, llm) -> AgentState:
    # Construct context prompt
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
    
    # Invoke LLM
    response = await llm.ainvoke([system_prompt, context_message])
    answer = response.content
    
    return {
        **state,
        "answer": answer,
        "current_step": "complete",
        "messages": state["messages"] + [AIMessage(content=answer)]
    }
```

---

## Extension Patterns

### Adding a Tool

1. **Server**: Implement tool function
2. **Server**: Register with `@mcp.tool()`
3. **Agent**: Create new node
4. **Agent**: Wire into graph with edges
5. **Agent**: Update state schema if needed

### Adding Checkpointing

```python
from langgraph.checkpoint.sqlite import SqliteSaver

async def build_graph_with_memory(mcp_client, llm):
    workflow = StateGraph(AgentState)
    # ... add nodes and edges ...
    
    # Create persistent checkpointer
    async with SqliteSaver.from_conn_string("./checkpoints.db") as memory:
        return workflow.compile(checkpointer=memory)

# In main():
config = {"configurable": {"thread_id": "user-123"}}
result = await graph.ainvoke(state, config=config)
```

**Benefits**:
- Conversation persists across sessions
- Can resume after interruption
- Enables human-in-the-loop workflows

### Human-in-the-Loop

```python
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["generate_answer"]  # Pause before final answer
)

# First invocation pauses at interrupt
result = await graph.ainvoke(state, config)
# result.next_node == "generate_answer"

# Human reviews intermediate state
print(f"About to answer with data: {result['weather_data']}")
approved = input("Approve? (y/n): ")

if approved == "y":
    # Resume execution
    final_result = await graph.invoke(None, config)  # Continue from checkpoint
```

---

## Performance Optimization

### Potential Improvements

1. **Parallel Tool Calls**:
   ```python
   # Current: Sequential
   # get_ip → resolve_location → fetch_weather
   
   # Optimized: Parallel where possible
   # If multiple weather sources:
   weather_results = await asyncio.gather(
       open_meteo.ainvoke(...),
       weatherapi.ainvoke(...),
       weather_gov.ainvoke(...)
   )
   ```

2. **Caching**:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000, ttl=3600)
   async def cached_ip_to_geo(ip: str):
       # IP → coords unlikely to change frequently
       return await get_location_from_ip(ip)
   ```

3. **Batch Processing**:
   ```python
   async def process_batch(questions: list[str]):
       tasks = [graph.ainvoke({"question": q, ...}) for q in questions]
       return await asyncio.gather(*tasks)
   ```

---

## Testing Strategy

### Unit Tests

Test individual nodes in isolation:

```python
@pytest.mark.asyncio
async def test_get_ip_node():
    # Mock tools
    mock_tool = Mock()
    mock_tool.ainvoke.return_value = "192.168.1.1"
    
    # Initial state
    state = {
        "question": "test",
        "public_ip": None,
        "messages": [],
        "error": None,
        "current_step": "started"
    }
    
    # Execute node
    result = await get_ip_node(state, [mock_tool])
    
    # Assertions
    assert result["public_ip"] == "192.168.1.1"
    assert result["error"] is None
    assert result["current_step"] == "ip_discovered"
```

### Integration Tests

Test graph execution end-to-end:

```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_workflow(running_mcp_server):
    client = MCPClient(url="http://localhost:8000/sse")
    llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")
    
    graph = await build_graph(client, llm)
    
    initial_state = {
        "question": "What is the weather forecast of the data center?",
        # ... initialize all fields
    }
    
    final_state = await graph.ainvoke(initial_state)
    
    assert final_state["answer"] is not None
    assert "temperature" in final_state["answer"].lower()
    assert final_state["error"] is None
```

---

## Monitoring & Observability

### LangSmith Integration

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT="datacenter-weather-prod"
```

**Captures**:
- Every LLM call with inputs/outputs
- Tool executions with latency
- State transitions
- Error traces
- Full execution graph visualization

### Custom Metrics

```python
import time

async def get_ip_node(state, tools):
    start_time = time.time()
    
    try:
        result = await tool.ainvoke({})
        duration = time.time() - start_time
        
        metrics.record("get_ip_duration", duration)
        metrics.increment("get_ip_success")
        
        return success_state
    except Exception as e:
        metrics.increment("get_ip_failure")
        raise
```

---

## Security Considerations

1. **API Key Management**: Never hardcode, use environment variables
2. **Input Validation**: All tool inputs validated server-side
3. **Rate Limiting**: Implement on MCP server for production
4. **Error Sanitization**: Don't leak sensitive data in error messages
5. **CORS**: Configure if exposing server to web clients
6. **Authentication**: Add auth layer for public deployments

---

This architecture balances:
- **Simplicity**: Clear, understandable structure
- **Robustness**: Multi-layer error handling
- **Extensibility**: Easy to add features
- **Observability**: Comprehensive logging and tracing
- **Performance**: Async throughout, ready for optimization

It represents **production best practices** for LangGraph agents in 2026.