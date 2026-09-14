## Aviationstack MCP Server

This project is an **MCP (Model Context Protocol) server** that provides a set of tools to interact with the [AviationStack API](https://aviationstack.com/). It exposes endpoints for retrieving real-time and future flight data, aircraft and airplane details, and core reference data (airports, airlines, routes, taxes), making it easy to integrate aviation data into your applications.

You can also find the Aviationstack MCP server in these well-known MCP server repositories for easy access:

- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers?tab=readme-ov-file#-community-servers)
- [punkpeye/awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers?tab=readme-ov-file#-travel--transportation)
- [glama](https://glama.ai/mcp/servers/@Pradumnasaraf/aviationstack-mcp)
- [smithery](https://smithery.ai/server/pradumnasaraf/aviationstack-mcp)
- [mcpservers.org](https://mcpservers.org/servers/pradumnasaraf/aviationstack-mcp)

### Demo

https://github.com/user-attachments/assets/9325fcce-8ecc-4b01-8923-4ccb2f6968f4

### Features

- **Look up a single flight by its flight number**
- **Get flights for a specific airline**
- **Fetch historical flights by date**
- **Retrieve arrival and departure schedules for airports**
- **Fetch future flight schedules**
- **Get random aircraft types**
- **Get detailed info on random airplanes**
- **Get detailed info on random countries**
- **Get detailed info on random cities**
- **List airports, airlines, routes, and taxes**

All endpoints are implemented as MCP tools and are ready to be used in an MCP-compatible environment.

Every tool returns the same JSON envelope. On success: `{"ok": true, "count": N, "data": [...]}`, plus a `pagination` block on the `list_*` tools and a `message` when there were no matches. On failure: `{"ok": false, "context": "...", "error": "..."}`. Any `limit` is capped at 100 records per call.

### Prerequisites

- Aviationstack API Key (You can get a FREE API Key from [Aviationstack](https://aviationstack.com/signup/free))
- Python 3.13 or newer
- uv package manager installed

### Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_flight_status(flight_iata: str, flight_date: str = "")` | Look up one flight by its IATA flight number, for today or a given date. | - **`flight_iata`**: Flight IATA number (e.g., "AA100")<br> - **`flight_date`**: Optional date in `YYYY-MM-DD` format |
| `flights_with_airline(airline_name: str, number_of_flights: int, flight_status: str = "")` | Get live flights for a specific airline. | - **`airline_name`**: Name of the airline (e.g., "Delta Air Lines")<br> - **`number_of_flights`**: Number of flights to return<br> - **`flight_status`**: Optional status filter: `scheduled`, `active`, `landed`, `cancelled`, `incident`, `diverted` |
| `historical_flights_by_date(flight_date: str, number_of_flights: int, airline_iata: str = "", dep_iata: str = "", arr_iata: str = "")` | Get historical flights for a date (Basic plan+). | - **`flight_date`**: Date in `YYYY-MM-DD` format<br> - **`number_of_flights`**: Number of flights to return<br> - **`airline_iata`**: Optional airline IATA filter<br> - **`dep_iata`**: Optional departure airport IATA filter<br> - **`arr_iata`**: Optional arrival airport IATA filter |
| `flight_arrival_departure_schedule(airport_iata_code: str, schedule_type: str, airline_name: str, number_of_flights: int)` | Get arrival or departure schedules for a given airport and airline. | - **`airport_iata_code`**: IATA code of the airport (e.g., "JFK")<br> - **`schedule_type`**: "arrival" or "departure"<br> - **`airline_name`**: Name of the airline<br> - **`number_of_flights`**: Number of flights to return |
| `future_flights_arrival_departure_schedule(airport_iata_code: str, schedule_type: str, airline_iata: str, date: str, number_of_flights: int)` | Get future scheduled flights for a given airport, airline, and date. | - **`airport_iata_code`** : IATA code of the airport<br> - **`schedule_type`**: "arrival" or "departure"<br> - **`airline_iata`**: IATA code of the airline (e.g., "DL" for Delta)<br> - **`date`**: Date in `YYYY-MM-DD` format<br> - **`number_of_flights`**: Number of flights to return |
| `random_aircraft_type(number_of_aircraft: int)` | Get aircraft types from a random offset in the dataset. | - **`number_of_aircraft`**: Number of aircraft types to return |
| `random_airplanes_detailed_info(number_of_airplanes: int)` | Get detailed info on airplanes from a random offset in the dataset. | - **`number_of_airplanes`**: Number of airplanes to return |
| `random_countries_detailed_info(number_of_countries: int)` | Get detailed info on countries from a random offset in the dataset. | - **`number_of_countries`**: Number of countries to return |
| `random_cities_detailed_info(number_of_cities: int)` | Get detailed info on cities from a random offset in the dataset. | - **`number_of_cities`**: Number of cities to return |
| `list_airports(limit: int = 10, offset: int = 0, search: str = "")` | List airports. | - **`limit`**: Number of results to return<br> - **`offset`**: Pagination offset<br> - **`search`**: Optional search query |
| `list_airlines(limit: int = 10, offset: int = 0, search: str = "")` | List airlines. | - **`limit`**: Number of results to return<br> - **`offset`**: Pagination offset<br> - **`search`**: Optional search query |
| `list_routes(limit: int = 10, offset: int = 0, airline_iata: str = "", dep_iata: str = "", arr_iata: str = "")` | List routes. | - **`limit`**: Number of results to return<br> - **`offset`**: Pagination offset<br> - **`airline_iata`**: Optional airline IATA filter<br> - **`dep_iata`**: Optional departure airport IATA filter<br> - **`arr_iata`**: Optional arrival airport IATA filter |
| `list_taxes(limit: int = 10, offset: int = 0, search: str = "")` | List aviation taxes. | - **`limit`**: Number of results to return<br> - **`offset`**: Pagination offset<br> - **`search`**: Optional search query |

### Prompts

The server ships reusable prompts that steer a model toward the right tool.

| Prompt | Arguments | Purpose |
|--------|-----------|---------|
| `plan_flight_status_lookup` | `flight_iata`, `flight_date` | Check one specific flight, and explain its codeshares. |
| `plan_airline_flight_lookup` | `airline_name`, `number_of_flights` | Query live flights for an airline. |
| `plan_future_schedule_lookup` | `airport_iata_code`, `date`, `schedule_type` | Query a future airport schedule. |
| `plan_reference_data_lookup` | `data_type`, `search` | Explore airport, airline, route or tax reference data. |

### Resources

| Resource | URI | Contents |
|----------|-----|----------|
| `server_metadata` | `aviationstack://meta/server` | API base URL and the accepted API key variables. |
| `aviationstack_endpoints` | `aviationstack://meta/endpoints` | The Aviationstack endpoints each tool calls. |
| `tool_input_examples` | `aviationstack://examples/tool-input/{tool_name}` | A sample payload for a given tool. |

### Development

- The main server logic is in `src/aviationstack_mcp/server.py`.
- All MCP tools are defined as Python functions decorated with `@mcp.tool()`.
- Each tool is a thin wrapper that validates input with a Pydantic model, then calls the
  matching plain function. Tests target the plain functions.
- The server uses the `FastMCP` class from `mcp.server.fastmcp`. The `mcp` dependency is
  pinned to `<2`, because 2.x renames `FastMCP` to `MCPServer`.
- Tools never raise. Every failure is caught and returned as the error envelope.

Set up and run the checks the CI runs:

```bash
uv sync --all-groups

# Unit tests
uv run python -m unittest discover -s tests -v

# Lint, must stay at 10.00/10
uv run pylint $(git ls-files '*.py')

# Coverage
uv run coverage run --source=aviationstack_mcp -m unittest discover -s tests
uv run coverage report
```

`.well-known/mcp/server-card.json` is generated, not hand-edited. After changing any tool,
prompt or resource, regenerate it or CI will fail:

```bash
uv run python scripts/generate_server_card.py          # rewrite the card
uv run python scripts/generate_server_card.py --check  # what CI runs
```

### MCP Server configuration

To add this server to your favorite MCP client, you can add the following to your MCP client configuration file.

1. Using `uvx` without cloning the repository (recommended)

```json
{
  "mcpServers": {
    "Aviationstack MCP": {
      "command": "uvx",
      "args": [
        "aviationstack-mcp"
      ],
      "env": {
        "AVIATION_STACK_API_KEY": "<your-api-key>"
      }
    }
  }
}
```

2. By cloning the repository and running the server locally

```json
{
  "mcpServers": {
    "Aviationstack MCP": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/aviationstack-mcp/src/aviationstack_mcp",
        "run",
        "-m",
        "aviationstack_mcp",
        "mcp",
        "run"
      ],
      "env": {
        "AVIATION_STACK_API_KEY": "<your-api-key>"
      }
    }
  }
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
