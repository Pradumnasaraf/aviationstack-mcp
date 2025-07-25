## Aviationstack MCP Server

This project is an **MCP (Model Context Protocol) server** that provides a set of tools to interact with the [AviationStack API](https://aviationstack.com/). It exposes endpoints for retrieving real-time and future flight data, aircraft types, and airplane details, making it easy to integrate aviation data into your applications.

### Demo

https://github.com/user-attachments/assets/9325fcce-8ecc-4b01-8923-4ccb2f6968f4

### Features

- **Get flights for a specific airline**
- **Retrieve arrival and departure schedules for airports**
- **Fetch future flight schedules**
- **Get random aircraft types**
- **Get detailed info on random airplanes**
- **Get detailed info on random countries**
- **Get detailed info on random cities**

All endpoints are implemented as MCP tools and are ready to be used in an MCP-compatible environment.

### Prerequisites

- Aviationstack API Key (You can get a FREE API Key from [Aviationstack](https://aviationstack.com/signup/free))
- Python 3.13 or newer
- uv package manager installed

### Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `flights_with_airline(airline_name: str, number_of_flights: int)` | Get a random sample of flights for a specific airline. | - **`airline_name`**: Name of the airline (e.g., "Delta Air Lines")<br> - **`number_of_flights`**: Number of flights to return |
| `flight_arrival_departure_schedule(airport_iata_code: str, schedule_type: str, airline_name: str, number_of_flights: int)` | Get arrival or departure schedules for a given airport and airline. | - **`airport_iata_code`**: IATA code of the airport (e.g., "JFK")<br> - **`schedule_type`**: "arrival" or "departure"<br> - **`airline_name`**: Name of the airline<br> - **`number_of_flights`**: Number of flights to return |
| `future_flights_arrival_departure_schedule(airport_iata_code: str, schedule_type: str, airline_iata: str, date: str, number_of_flights: int)` | Get future scheduled flights for a given airport, airline, and date. | - **`airport_iata_code`** : IATA code of the airport<br> - **`schedule_type`**: "arrival" or "departure"<br> - **`airline_iata`**: IATA code of the airline (e.g., "DL" for Delta)<br> - **`date`**: Date in `YYYY-MM-DD` format<br> - **`number_of_flights`**: Number of flights to return |
| `random_aircraft_type(number_of_aircraft: int)` | Get random aircraft types. | - **`number_of_aircraft`**: Number of aircraft types to return |
| `random_airplanes_detailed_info(number_of_airplanes: int)` | Get detailed info on random airplanes. | - **`number_of_airplanes`**: Number of airplanes to return |
| `random_countries_detailed_info(number_of_countries: int)` | Get detailed info on random countries. | - **`number_of_countries`**: Number of countries to return |
| `random_cities_detailed_info(number_of_cities: int)` | Get detailed info on random cities. | - **`number_of_cities`**: Number of cities to return |

### Development

- The main server logic is in `server.py`.
- All MCP tools are defined as Python functions decorated with `@mcp.tool()`.
- The server uses the `FastMCP` class from `mcp.server.fastmcp`.

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
