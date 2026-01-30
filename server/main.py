from mcp.server.fastmcp import FastMCP
from server.tools import get_public_ip, get_location_from_ip, get_weather_forecast

# Initialize FastMCP server
mcp = FastMCP("Datacenter Weather Tools")

@mcp.tool()
async def ipify() -> str:
    #
    """It gets the public IP address of the server. Returns the IP as a string."""
    return await get_public_ip()
@mcp.tool()
async def ip_to_geo(ip: str) -> str:
    """
       It takes an IP address as input and returns its geographical location (latitude and longitude).
       Args: ip (str): The IP address to look up.
       Returns: str: A string containing the latitude and longitude.
    """
    location = await get_location_from_ip(ip)
    return f"{location['latitude']}, {location['longitude']}"

@mcp.tool()
async def weather_forecast(latitude: float, longitude: float) -> str:
    """
    It takes latitude and longitude as input and returns the current weather forecast for that location.
    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.
    Returns:
        str: A string describing the current weather forecast.
    """
    return await get_weather_forecast(latitude, longitude)
# Start the MCP server
if __name__ == "__main__":
    # Run the MCP server with Server-Sent Events (SSE) transport on the default port 8000  
    mcp.run(transport="sse")

