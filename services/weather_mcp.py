from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_weather():
    """
    Returns the weather for the current location.
    """
    return {"weather": "sunny"}

if __name__ == "__main__":
    mcp.run(transport="streamable-http")