"""Aviationstack MCP server tools.

Note: Ensure 'requests' and 'mcp' packages are installed and importable in your environment.
"""
import os
import json
import random
import requests
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Aviationstack MCP")

def fetch_flight_data(url: str, params: dict) -> dict:
    """Fetch flight data from the AviationStack API."""
    api_key = os.getenv('AVIATION_STACK_API_KEY')
    if not api_key:
        raise ValueError("AVIATION_STACK_API_KEY not set in environment.")
    params = {'access_key': api_key, **params}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

@mcp.tool()
def flights_with_airline(airline_name: str, number_of_flights: int) -> str:
    """MCP tool to get flights with a specific airline."""
    try:
        data = fetch_flight_data(
            'http://api.aviationstack.com/v1/flights',
            {'airline_name': airline_name}
        )
        filtered_flights = []
        data_list = data.get('data', [])
        number_of_flights = min(number_of_flights, len(data_list))

        # Sample random flights from the data list
        sampled_flights = random.sample(data_list, number_of_flights)

        for flight in sampled_flights:
            filtered_flights.append({
                'flight_number': flight.get('flight').get('iata'),
                'airline': flight.get('airline').get('name'),
                'departure_airport': flight.get('departure').get('airport'),
                'departure_timezone': flight.get('departure').get('timezone'),
                'departure_time': flight.get('departure').get('scheduled'),
                'arrival_airport': flight.get('arrival').get('airport'),
                'arrival_timezone': flight.get('arrival').get('timezone'),
                'flight_status': flight.get('flight_status'),
                'departure_delay': flight.get('departure').get('delay'),
                'departure_terminal': flight.get('departure').get('terminal'),
                'departure_gate': flight.get('departure').get('gate'),
            })
        return json.dumps(filtered_flights) if filtered_flights else (
            f"No flights found for airline '{airline_name}'."
        )
    except requests.RequestException as e:
        return f"Request error: {str(e)}"
    except (KeyError, ValueError, TypeError) as e:
        return f"Error fetching flights: {str(e)}"

@mcp.tool()
def flight_arrival_departure_schedule(
    airport_iata_code: str,
    schedule_type: str,
    airline_name: str,
    number_of_flights: int
) -> str:
    """MCP tool to get flight arrival and departure schedule."""
    try:
        data = fetch_flight_data(
            'http://api.aviationstack.com/v1/timetable',
            {'iataCode': airport_iata_code, 'type': schedule_type, 'airline_name': airline_name}
        )
        data_list = data.get('data', [])
        number_of_flights = min(number_of_flights, len(data_list))

        # Sample random flights from the data list
        sampled_flights = random.sample(data_list, number_of_flights)

        filtered_flights = []
        for flight in sampled_flights:
            filtered_flights.append({
                'airline': flight.get('airline').get('name'),
                'flight_number': flight.get('flight').get('iataNumber'),
                'departure_estimated_time': flight.get('departure').get('estimatedTime'),
                'departure_scheduled_time': flight.get('departure').get('scheduledTime'),
                'departure_actual_time': flight.get('departure').get('actualTime'),
                'departure_terminal': flight.get('departure').get('terminal'),
                'departure_gate': flight.get('departure').get('gate'),
                'arrival_estimated_time': flight.get('arrival').get('estimatedTime'),
                'arrival_scheduled_time': flight.get('arrival').get('scheduledTime'),
                'arrival_airport_code': flight.get('arrival').get('iataCode'),
                'arrival_terminal': flight.get('arrival').get('terminal'),
                'arrival_gate': flight.get('arrival').get('gate'),
                'departure_delay': flight.get('departure').get('delay'),
            })
        return json.dumps(filtered_flights) if filtered_flights else (
            f"No flights found for iata code '{airport_iata_code}'."
        )
    except requests.RequestException as e:
        return f"Request error: {str(e)}"
    except (KeyError, ValueError, TypeError) as e:
        return f"Error fetching flight schedule: {str(e)}"

@mcp.tool()
def future_flights_arrival_departure_schedule(
    airport_iata_code: str,
    schedule_type: str,
    airline_iata: str,
    date: str,
    number_of_flights: int
) -> str:
    """MCP tool to get flight future arrival and departure schedule."""
    try:
        data = fetch_flight_data(
            'http://api.aviationstack.com/v1/flightsFuture',
            {
                'iataCode': airport_iata_code,
                'type': schedule_type,
                'airline_iata': airline_iata,
                'date': date
            }
        )  # date is in format YYYY-MM-DD
        data_list = data.get('data', [])
        number_of_flights = min(number_of_flights, len(data_list))

        # Sample random flights from the data list
        sampled_flights = random.sample(data_list, number_of_flights)

        filtered_flights = []

        for flight in sampled_flights:
            filtered_flights.append({
                'airline': flight.get('airline').get('name'),
                'flight_number': flight.get('flight').get('iataNumber'),
                'departure_scheduled_time': flight.get('departure').get('scheduledTime'),
                'arrival_scheduled_time': flight.get('arrival').get('scheduledTime'),
                'arrival_airport_code': flight.get('arrival').get('iataCode'),
                'arrival_terminal': flight.get('arrival').get('terminal'),
                'arrival_gate': flight.get('arrival').get('gate'),
                'aircraft': flight.get('aircraft').get('modelText')
            })
        return json.dumps(filtered_flights) if filtered_flights else (
            f"No flights found for iata code '{airport_iata_code}'."
        )
    except requests.RequestException as e:
        return f"Request error: {str(e)}"
    except (KeyError, ValueError, TypeError) as e:
        return f"Error fetching flight future schedule: {str(e)}"

@mcp.tool()
def random_aircraft_type(number_of_aircraft: int) -> str:
    """MCP tool to get random aircraft type."""
    try:
        data = fetch_flight_data('http://api.aviationstack.com/v1/aircraft_types', {})
        data_list = data.get('data', [])
        number_of_aircraft = min(number_of_aircraft, len(data_list))

        # Sample random aircraft types from the data list
        sampled_aircraft_types = random.sample(data_list, number_of_aircraft)

        aircraft_types = []
        for aircraft_type in sampled_aircraft_types:
            aircraft_types.append({
                'aircraft_name': aircraft_type.get('aircraft_name'),
                'icao_code': aircraft_type.get('iata_code'),
            })
        return json.dumps(aircraft_types)
    except requests.RequestException as e:
        return f"Request error: {str(e)}"
    except (KeyError, ValueError, TypeError) as e:
        return f"Error fetching aircraft type: {str(e)}"

@mcp.tool()
def random_airplanes_detailed_info(number_of_airplanes: int) -> str:
    """MCP tool to get random airplanes."""
    try:
        data = fetch_flight_data('http://api.aviationstack.com/v1/airplanes', {})
        data_list = data.get('data', [])
        number_of_airplanes = min(number_of_airplanes, len(data_list))

        # Sample random airplanes from the data list
        sampled_airplanes = random.sample(data_list, number_of_airplanes)

        airplanes = []
        for airplane in sampled_airplanes:
            airplanes.append({
                'production_line': airplane.get('production_line'),
                'plane_owner': airplane.get('plane_owner'),
                'plane_age': airplane.get('plane_age'),
                'model_name': airplane.get('model_name'),
                'plane_series': airplane.get('plane_series'),
                'registration_number': airplane.get('registration_number'),
                'engines_type': airplane.get('engines_type'),
                'engines_count': airplane.get('engines_count'),
                'delivery_date': airplane.get('delivery_date'),
                'first_flight_date': airplane.get('first_flight_date'),
            })
        return json.dumps(airplanes)
    except requests.RequestException as e:
        return f"Request error: {str(e)}"
    except (KeyError, ValueError, TypeError) as e:
        return f"Error fetching airplanes: {str(e)}"

def main():
    """Run the MCP server."""
    mcp.run()

if __name__ == "__main__":
    main()
