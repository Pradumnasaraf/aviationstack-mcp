"""aviationstack_mcp package initialization and main entry point."""
from .server import mcp

def main() -> None:
    """Run the MCP server."""
    mcp.run()
