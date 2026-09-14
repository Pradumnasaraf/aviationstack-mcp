"""Generate .well-known/mcp/server-card.json from the running server definition.

The card is a discovery document, so it has to match the tools, resources and
prompts the server actually exposes. Generating it keeps the two from drifting.
Run with --check in CI to fail when the committed card is stale.
"""

import argparse
import asyncio
import json
import sys
import tomllib
from pathlib import Path

from aviationstack_mcp.server import CONFIG_SCHEMA, mcp

REPO_ROOT = Path(__file__).resolve().parent.parent
CARD_PATH = REPO_ROOT / ".well-known" / "mcp" / "server-card.json"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

DESCRIPTION = (
    "MCP server for Aviationstack flight, schedule, and aviation reference data."
)


def _project_version() -> str:
    """Read the package version so the card never lags behind pyproject."""
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


async def _build_card() -> dict:
    """Introspect the server and assemble the card payload."""
    tools = await mcp.list_tools()
    resources = await mcp.list_resources()
    templates = await mcp.list_resource_templates()
    prompts = await mcp.list_prompts()

    resource_entries = [
        {
            "name": resource.name,
            "uri": str(resource.uri),
            "description": resource.description,
            "mimeType": resource.mimeType,
        }
        for resource in resources
    ]
    resource_entries += [
        {
            "name": template.name,
            "uriTemplate": template.uriTemplate,
            "description": template.description,
        }
        for template in templates
    ]

    return {
        "serverInfo": {
            "name": "aviationstack-mcp",
            "title": "Aviationstack MCP",
            "version": _project_version(),
            "description": DESCRIPTION,
            "configSchema": CONFIG_SCHEMA,
        },
        "tools": [tool.model_dump(exclude_none=True) for tool in tools],
        "resources": resource_entries,
        "prompts": [prompt.model_dump(exclude_none=True) for prompt in prompts],
    }


def main() -> int:
    """Write the card, or verify the committed one is current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed card differs from the generated one.",
    )
    args = parser.parse_args()

    rendered = json.dumps(asyncio.run(_build_card()), indent=2) + "\n"

    if args.check:
        current = CARD_PATH.read_text(encoding="utf-8") if CARD_PATH.exists() else ""
        if current != rendered:
            print(
                f"{CARD_PATH.relative_to(REPO_ROOT)} is out of date. "
                "Run: python scripts/generate_server_card.py",
                file=sys.stderr,
            )
            return 1
        print(f"{CARD_PATH.relative_to(REPO_ROOT)} is up to date.")
        return 0

    CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    CARD_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {CARD_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
