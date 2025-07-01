# Aviationstack MCP Server

This project is an **MCP (Multi-Channel Platform) server** that provides a set of tools to interact with the [AviationStack API](https://aviationstack.com/). It exposes endpoints for retrieving real-time and future flight data, aircraft types, and airplane details, making it easy to integrate aviation data into your applications.

## Features

- **Get flights for a specific airline**
- **Retrieve arrival and departure schedules for airports**
- **Fetch future flight schedules**
- **Get random aircraft types**
- **Get detailed info on random airplanes**

All endpoints are implemented as MCP tools and are ready to be used in an MCP-compatible environment.

### Prerequisites

- Aviationstack API Key (You can get a FREE API Key from [Aviationstack](https://aviationstack.com/signup/free))
- Python 3.10 or newer
- uv package manager 

## Available Tools

#### 1. `flights_with_airline(airline_name: str, number_of_flights: int)`

Get a random sample of flights for a specific airline.

- **Parameters:**
  - `airline_name`: Name of the airline (e.g., "Delta Air Lines")
  - `number_of_flights`: Number of flights to return

#### 2. `flight_arrival_departure_schedule(airport_iata_code: str, type: str, airline_name: str, number_of_flights: int)`

Get arrival or departure schedules for a given airport and airline.

- **Parameters:**
  - `airport_iata_code`: IATA code of the airport (e.g., "JFK")
  - `type`: "arrival" or "departure"
  - `airline_name`: Name of the airline
  - `number_of_flights`: Number of flights to return

#### 3. `future_flights_arrival_departure_schedule(airport_iata_code: str, type: str, airline_iata: str, date: str, number_of_flights: int)`

Get future scheduled flights for a given airport, airline, and date.

- **Parameters:**
  - `airport_iata_code`: IATA code of the airport
  - `type`: "arrival" or "departure"
  - `airline_iata`: IATA code of the airline (e.g., "DL" for Delta)
  - `date`: Date in `YYYY-MM-DD` format
  - `number_of_flights`: Number of flights to return

#### 4. `random_aircraft_type(number_of_aircraft: int)`

Get random aircraft types.

- **Parameters:**
  - `number_of_aircraft`: Number of aircraft types to return

### 5. `random_airplanes_detailed_info(number_of_airplanes: int)`

Get detailed info on random airplanes.

- **Parameters:**
  - `number_of_airplanes`: Number of airplanes to return


## Development

- The main server logic is in `server.py`.
- All MCP tools are defined as Python functions decorated with `@mcp.tool()`.
- The server uses the `FastMCP` class from `mcp.server.fastmcp`.

### MCP Server configuration

To add this server to your favorite MCP client, you can add the following to your MCP client configuration file.

```json
{
  "mcpServers": {
    "Aviationstack MCP": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp[cli],requests",
        "mcp",
        "run",
        "/path/to/your/server.py"
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