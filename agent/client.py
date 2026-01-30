import contextlib
import httpx
from typing import Optional
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from langchain_core.tools import StructuredTool

class MCPClient:
    def __init__(self, url: str = "http://localhost:8000/sse"):
        self.url = url
        self.session: Optional[ClientSession] = None
        self._exit_stack = contextlib.AsyncExitStack()

    async def __aenter__(self):
        # Establish the SSE connection
        sse_transport = await self._exit_stack.enter_async_context(
            sse_client(self.url)
        )
        read_stream, write_stream = sse_transport
        
        # Initialize the MCP Session over that transport
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        # Initialize the connection
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exit_stack:
            await self._exit_stack.aclose()

    async def get_tools(self):
        """Discovers tools from the server and converts them to LangChain tools."""
        if not self.session:
            raise RuntimeError("Client not connected. Use 'async with client:' context.")
            
        result = await self.session.list_tools()
        tools = []
        
        for mcp_tool in result.tools:
            tools.append(self._create_langchain_tool(mcp_tool))
            
        return tools

    def _create_langchain_tool(self, mcp_tool):
        """Helper to convert an MCP tool definition to a LangChain StructuredTool"""
        from pydantic import create_model, Field

        # Dynamically create the Pydantic model for args_schema
        fields = {}
        if mcp_tool.inputSchema and "properties" in mcp_tool.inputSchema:
            for name, schema in mcp_tool.inputSchema["properties"].items():
                # Default type is string if not specified
                python_type = str
                if schema.get("type") == "number":
                    python_type = float
                elif schema.get("type") == "integer":
                    python_type = int
                elif schema.get("type") == "boolean":
                    python_type = bool
                
                # Check if required
                is_required = name in mcp_tool.inputSchema.get("required", [])
                default = ... if is_required else None
                
                description = schema.get("description", "")
                fields[name] = (python_type, Field(default=default, description=description))
        
        # Create the Pydantic model
        ArgsModel = create_model(f"{mcp_tool.name}Arguments", **fields)

        async def _tool_wrapper(**kwargs):
            # Here is the execution logic when the LLM calls this tool
            assert self.session is not None
            result = await self.session.call_tool(mcp_tool.name, arguments=kwargs)
            
            # Validate response
            if not result or not hasattr(result, 'content'):
                raise RuntimeError(f"Tool {mcp_tool.name} returned invalid response structure")
            
            # MCP returns a list of content items (TextContent or ImageContent)
            # We join the text content to return a single string to the LLM
            text_content = [c.text for c in result.content if c.type == "text"]
            
            if not text_content:
                raise RuntimeError(f"Tool {mcp_tool.name} returned no text content")
            
            return "\n".join(text_content)

        # Create the structured tool with schema
        return StructuredTool.from_function(
            func=None,
            coroutine=_tool_wrapper,
            name=mcp_tool.name,
            description=mcp_tool.description,
            args_schema=ArgsModel
        )
