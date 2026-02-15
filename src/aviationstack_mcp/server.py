"""Aviationstack MCP server tools."""

import json
import os
import random
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any

import requests
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "aviationstack_api_key": {
            "type": "string",
            "description": "Aviationstack API key used to authenticate API requests.",
            "minLength": 1,
        }
    },
    "required": ["aviationstack_api_key"],
    "additionalProperties": False,
}

mcp = FastMCP(
    "Aviationstack MCP",
    instructions=(
        "Use these tools to fetch real-time, historical, and reference aviation data from "
        "Aviationstack. Provide IATA/ICAO codes when available and request small limits first."
    ),
    config_schema=CONFIG_SCHEMA,
)

API_BASE_URL = "https://api.aviationstack.com/v1"


class FlightsWithAirlineInput(BaseModel):
    """Input schema for flights_with_airline tool."""

    model_config = ConfigDict(extra="forbid")

    airline_name: str = Field(
        ...,
        description="Airline name to filter flights (for example: Delta Air Lines).",
        min_length=1,
    )
    number_of_flights: int = Field(
        ...,
        description="Number of random flights to return.",
        gt=0,
    )


class HistoricalFlightsByDateInput(BaseModel):
    """Input schema for historical_flights_by_date tool."""

    model_config = ConfigDict(extra="forbid")

    flight_date: str = Field(
        ...,
        description="Date in YYYY-MM-DD format.",
        examples=["2026-03-01"],
    )
    number_of_flights: int = Field(
        ...,
        description="Number of random flights to return.",
        gt=0,
    )
    airline_iata: str = Field(
        default="",
        description="Optional airline IATA code filter (for example: DL).",
    )
    dep_iata: str = Field(
        default="",
        description="Optional departure airport IATA code filter (for example: JFK).",
    )
    arr_iata: str = Field(
        default="",
        description="Optional arrival airport IATA code filter (for example: LAX).",
    )


class FlightArrivalDepartureScheduleInput(BaseModel):
    """Input schema for flight_arrival_departure_schedule tool."""

    model_config = ConfigDict(extra="forbid")

    airport_iata_code: str = Field(
        ...,
        description="Airport IATA code (for example: SFO).",
        min_length=1,
    )
    schedule_type: str = Field(
        ...,
        description="Schedule type: arrival or departure.",
        examples=["arrival", "departure"],
    )
    airline_name: str = Field(
        default="",
        description="Optional airline name filter.",
    )
    number_of_flights: int = Field(
        ...,
        description="Number of random flights to return.",
        gt=0,
    )


class FutureFlightsArrivalDepartureScheduleInput(BaseModel):
    """Input schema for future_flights_arrival_departure_schedule tool."""

    model_config = ConfigDict(extra="forbid")

    airport_iata_code: str = Field(
        ...,
        description="Airport IATA code (for example: SFO).",
        min_length=1,
    )
    schedule_type: str = Field(
        ...,
        description="Schedule type: arrival or departure.",
        examples=["arrival", "departure"],
    )
    airline_iata: str = Field(
        default="",
        description="Optional airline IATA code filter (for example: UA).",
    )
    date: str = Field(
        ...,
        description="Future date in YYYY-MM-DD format.",
        examples=["2026-03-01"],
    )
    number_of_flights: int = Field(
        ...,
        description="Number of random flights to return.",
        gt=0,
    )


class RandomAircraftTypeInput(BaseModel):
    """Input schema for random_aircraft_type tool."""

    model_config = ConfigDict(extra="forbid")

    number_of_aircraft: int = Field(
        ...,
        description="Number of random aircraft types to return.",
        gt=0,
    )


class RandomAirplanesDetailedInfoInput(BaseModel):
    """Input schema for random_airplanes_detailed_info tool."""

    model_config = ConfigDict(extra="forbid")

    number_of_airplanes: int = Field(
        ...,
        description="Number of random airplanes to return.",
        gt=0,
    )


class RandomCountriesDetailedInfoInput(BaseModel):
    """Input schema for random_countries_detailed_info tool."""

    model_config = ConfigDict(extra="forbid")

    number_of_countries: int = Field(
        ...,
        description="Number of random countries to return.",
        gt=0,
    )


class RandomCitiesDetailedInfoInput(BaseModel):
    """Input schema for random_cities_detailed_info tool."""

    model_config = ConfigDict(extra="forbid")

    number_of_cities: int = Field(
        ...,
        description="Number of random cities to return.",
        gt=0,
    )


class ListAirportsInput(BaseModel):
    """Input schema for list_airports tool."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(
        default=10,
        description="Maximum number of airports to return.",
        gt=0,
    )
    offset: int = Field(
        default=0,
        description="Offset for pagination.",
        ge=0,
    )
    search: str = Field(
        default="",
        description="Optional airport search text for autocomplete.",
    )


class ListAirlinesInput(BaseModel):
    """Input schema for list_airlines tool."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(
        default=10,
        description="Maximum number of airlines to return.",
        gt=0,
    )
    offset: int = Field(
        default=0,
        description="Offset for pagination.",
        ge=0,
    )
    search: str = Field(
        default="",
        description="Optional airline search text for autocomplete.",
    )


class ListRoutesInput(BaseModel):
    """Input schema for list_routes tool."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(
        default=10,
        description="Maximum number of routes to return.",
        gt=0,
    )
    offset: int = Field(
        default=0,
        description="Offset for pagination.",
        ge=0,
    )
    airline_iata: str = Field(
        default="",
        description="Optional airline IATA code filter.",
    )
    dep_iata: str = Field(
        default="",
        description="Optional departure airport IATA code filter.",
    )
    arr_iata: str = Field(
        default="",
        description="Optional arrival airport IATA code filter.",
    )


class ListTaxesInput(BaseModel):
    """Input schema for list_taxes tool."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(
        default=10,
        description="Maximum number of tax records to return.",
        gt=0,
    )
    offset: int = Field(
        default=0,
        description="Offset for pagination.",
        ge=0,
    )
    search: str = Field(
        default="",
        description="Optional tax search text.",
    )


def _safe_get(obj: dict[str, Any] | None, *keys: str) -> Any:
    """Safely read nested keys from dictionaries."""
    current: Any = obj
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _validate_positive_int(value: int, param_name: str) -> None:
    if value <= 0:
        raise ValueError(f"'{param_name}' must be greater than 0.")


def _validate_non_negative_int(value: int, param_name: str) -> None:
    if value < 0:
        raise ValueError(f"'{param_name}' must be 0 or greater.")


def _validate_iso_date(date_value: str, param_name: str) -> None:
    """Validate strict YYYY-MM-DD date format."""
    try:
        parsed = datetime.strptime(date_value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"'{param_name}' must be in YYYY-MM-DD format.") from exc
    if parsed.strftime("%Y-%m-%d") != date_value:
        raise ValueError(f"'{param_name}' must be in YYYY-MM-DD format.")


def _error_response(context: str, exc: Exception) -> str:
    return json.dumps({"ok": False, "context": context, "error": str(exc)})


def _sample_data(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    _validate_positive_int(count, "count")
    number_to_fetch = min(count, len(items))
    return random.sample(items, number_to_fetch)


def _get_api_key() -> str:
    api_key = (
        os.getenv("AVIATION_STACK_API_KEY")
        or os.getenv("AVIATIONSTACK_API_KEY")
        or os.getenv("aviationstack_api_key")
    )
    if not api_key:
        raise ValueError(
            "Aviationstack API key not set. Provide AVIATION_STACK_API_KEY or "
            "aviationstack_api_key."
        )
    return api_key


def fetch_flight_data(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fetch data from the Aviationstack API."""
    api_key = _get_api_key()
    request_params = {"access_key": api_key, **params}
    response = requests.get(f"{API_BASE_URL}/{endpoint}", params=request_params, timeout=15)
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if payload.get("error"):
        error = payload["error"]
        code = error.get("code", "api_error")
        error_type = error.get("type", "api_error")
        message = error.get("message", "Unknown API error")
        raise ValueError(f"{error_type} ({code}): {message}")

    response.raise_for_status()
    return payload


def _list_reference_data(
    endpoint: str,
    params: dict[str, Any],
    mapper: Callable[[dict[str, Any]], dict[str, Any]],
) -> str:
    data = fetch_flight_data(endpoint, params)
    return json.dumps([mapper(item) for item in data.get("data", [])])


def flights_with_airline(airline_name: str, number_of_flights: int) -> str:
    """Get a random sample of real-time flights for an airline."""
    try:
        _validate_positive_int(number_of_flights, "number_of_flights")
        data = fetch_flight_data(
            "flights", {"airline_name": airline_name, "limit": number_of_flights}
        )
        sampled_flights = _sample_data(data.get("data", []), number_of_flights)

        filtered_flights = []
        for flight in sampled_flights:
            filtered_flights.append(
                {
                    "flight_number": _safe_get(flight, "flight", "iata"),
                    "airline": _safe_get(flight, "airline", "name"),
                    "departure_airport": _safe_get(flight, "departure", "airport"),
                    "departure_timezone": _safe_get(flight, "departure", "timezone"),
                    "departure_time": _safe_get(flight, "departure", "scheduled"),
                    "arrival_airport": _safe_get(flight, "arrival", "airport"),
                    "arrival_timezone": _safe_get(flight, "arrival", "timezone"),
                    "flight_status": flight.get("flight_status"),
                    "departure_delay": _safe_get(flight, "departure", "delay"),
                    "departure_terminal": _safe_get(flight, "departure", "terminal"),
                    "departure_gate": _safe_get(flight, "departure", "gate"),
                }
            )
        if not filtered_flights:
            return f"No flights found for airline '{airline_name}'."
        return json.dumps(filtered_flights)
    except requests.RequestException as exc:
        return _error_response("fetching flights", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching flights", exc)


def historical_flights_by_date(
    flight_date: str,
    number_of_flights: int,
    airline_iata: str = "",
    dep_iata: str = "",
    arr_iata: str = "",
) -> str:
    """Get a random sample of historical flights for a specific date (Basic plan+)."""
    try:
        _validate_positive_int(number_of_flights, "number_of_flights")
        _validate_iso_date(flight_date, "flight_date")
        params: dict[str, Any] = {"flight_date": flight_date, "limit": number_of_flights}
        if airline_iata:
            params["airline_iata"] = airline_iata
        if dep_iata:
            params["dep_iata"] = dep_iata
        if arr_iata:
            params["arr_iata"] = arr_iata

        data = fetch_flight_data("flights", params)
        sampled_flights = _sample_data(data.get("data", []), number_of_flights)

        historical_flights = []
        for flight in sampled_flights:
            historical_flights.append(
                {
                    "flight_date": flight.get("flight_date"),
                    "flight_status": flight.get("flight_status"),
                    "flight_number": _safe_get(flight, "flight", "iata"),
                    "airline": _safe_get(flight, "airline", "name"),
                    "departure_airport": _safe_get(flight, "departure", "airport"),
                    "departure_time": _safe_get(flight, "departure", "scheduled"),
                    "arrival_airport": _safe_get(flight, "arrival", "airport"),
                    "arrival_time": _safe_get(flight, "arrival", "scheduled"),
                }
            )
        if not historical_flights:
            return f"No historical flights found for date '{flight_date}'."
        return json.dumps(historical_flights)
    except requests.RequestException as exc:
        return _error_response("fetching historical flights", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching historical flights", exc)


def flight_arrival_departure_schedule(
    airport_iata_code: str,
    schedule_type: str,
    airline_name: str,
    number_of_flights: int,
) -> str:
    """Get a random sample of current-day arrival/departure schedules for an airport."""
    try:
        _validate_positive_int(number_of_flights, "number_of_flights")
        normalized_schedule_type = schedule_type.lower()
        if normalized_schedule_type not in {"arrival", "departure"}:
            raise ValueError("'schedule_type' must be either 'arrival' or 'departure'.")

        params: dict[str, Any] = {"iataCode": airport_iata_code, "type": normalized_schedule_type}
        if airline_name:
            params["airline_name"] = airline_name

        data = fetch_flight_data("timetable", params)
        sampled_flights = _sample_data(data.get("data", []), number_of_flights)

        filtered_flights = []
        for flight in sampled_flights:
            filtered_flights.append(
                {
                    "airline": _safe_get(flight, "airline", "name"),
                    "flight_number": _safe_get(flight, "flight", "iataNumber"),
                    "departure_estimated_time": _safe_get(
                        flight, "departure", "estimatedTime"
                    ),
                    "departure_scheduled_time": _safe_get(
                        flight, "departure", "scheduledTime"
                    ),
                    "departure_actual_time": _safe_get(flight, "departure", "actualTime"),
                    "departure_terminal": _safe_get(flight, "departure", "terminal"),
                    "departure_gate": _safe_get(flight, "departure", "gate"),
                    "arrival_estimated_time": _safe_get(flight, "arrival", "estimatedTime"),
                    "arrival_scheduled_time": _safe_get(flight, "arrival", "scheduledTime"),
                    "arrival_airport_code": _safe_get(flight, "arrival", "iataCode"),
                    "arrival_terminal": _safe_get(flight, "arrival", "terminal"),
                    "arrival_gate": _safe_get(flight, "arrival", "gate"),
                    "departure_delay": _safe_get(flight, "departure", "delay"),
                }
            )
        if not filtered_flights:
            return f"No flights found for iata code '{airport_iata_code}'."
        return json.dumps(filtered_flights)
    except requests.RequestException as exc:
        return _error_response("fetching flight schedule", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching flight schedule", exc)


def future_flights_arrival_departure_schedule(
    airport_iata_code: str,
    schedule_type: str,
    airline_iata: str,
    date: str,
    number_of_flights: int,
) -> str:
    """Get a random sample of future flights for an airport and date."""
    try:
        _validate_positive_int(number_of_flights, "number_of_flights")
        _validate_iso_date(date, "date")
        normalized_schedule_type = schedule_type.lower()
        if normalized_schedule_type not in {"arrival", "departure"}:
            raise ValueError("'schedule_type' must be either 'arrival' or 'departure'.")

        params: dict[str, Any] = {
            "iataCode": airport_iata_code,
            "type": normalized_schedule_type,
            "date": date,
        }
        if airline_iata:
            params["airline_iata"] = airline_iata

        data = fetch_flight_data("flightsFuture", params)
        sampled_flights = _sample_data(data.get("data", []), number_of_flights)

        filtered_flights = []
        for flight in sampled_flights:
            filtered_flights.append(
                {
                    "airline": _safe_get(flight, "airline", "name"),
                    "flight_number": _safe_get(flight, "flight", "iataNumber"),
                    "departure_scheduled_time": _safe_get(
                        flight, "departure", "scheduledTime"
                    ),
                    "arrival_scheduled_time": _safe_get(flight, "arrival", "scheduledTime"),
                    "arrival_airport_code": _safe_get(flight, "arrival", "iataCode"),
                    "arrival_terminal": _safe_get(flight, "arrival", "terminal"),
                    "arrival_gate": _safe_get(flight, "arrival", "gate"),
                    "aircraft": _safe_get(flight, "aircraft", "modelText"),
                }
            )
        if not filtered_flights:
            return f"No flights found for iata code '{airport_iata_code}'."
        return json.dumps(filtered_flights)
    except requests.RequestException as exc:
        return _error_response("fetching flight future schedule", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching flight future schedule", exc)


def random_aircraft_type(number_of_aircraft: int) -> str:
    """Get random aircraft types."""
    try:
        _validate_positive_int(number_of_aircraft, "number_of_aircraft")
        data = fetch_flight_data("aircraft_types", {"limit": number_of_aircraft})
        sampled_aircraft_types = _sample_data(data.get("data", []), number_of_aircraft)

        aircraft_types = []
        for aircraft_type in sampled_aircraft_types:
            aircraft_types.append(
                {
                    "aircraft_name": aircraft_type.get("aircraft_name"),
                    "iata_code": aircraft_type.get("iata_code"),
                }
            )
        return json.dumps(aircraft_types)
    except requests.RequestException as exc:
        return _error_response("fetching aircraft type", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching aircraft type", exc)


def random_airplanes_detailed_info(number_of_airplanes: int) -> str:
    """Get detailed info for random airplanes."""
    try:
        _validate_positive_int(number_of_airplanes, "number_of_airplanes")
        data = fetch_flight_data("airplanes", {"limit": number_of_airplanes})
        sampled_airplanes = _sample_data(data.get("data", []), number_of_airplanes)

        airplanes = []
        for airplane in sampled_airplanes:
            airplanes.append(
                {
                    "production_line": airplane.get("production_line"),
                    "plane_owner": airplane.get("plane_owner"),
                    "plane_age": airplane.get("plane_age"),
                    "model_name": airplane.get("model_name"),
                    "model_code": airplane.get("model_code"),
                    "plane_series": airplane.get("plane_series"),
                    "registration_number": airplane.get("registration_number"),
                    "engines_type": airplane.get("engines_type"),
                    "engines_count": airplane.get("engines_count"),
                    "delivery_date": airplane.get("delivery_date"),
                    "first_flight_date": airplane.get("first_flight_date"),
                }
            )
        return json.dumps(airplanes)
    except requests.RequestException as exc:
        return _error_response("fetching airplanes", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching airplanes", exc)


def random_countries_detailed_info(number_of_countries: int) -> str:
    """Get detailed info for random countries."""
    try:
        _validate_positive_int(number_of_countries, "number_of_countries")
        data = fetch_flight_data("countries", {"limit": number_of_countries})
        sampled_countries = _sample_data(data.get("data", []), number_of_countries)

        countries = []
        for country in sampled_countries:
            countries.append(
                {
                    "country_name": country.get("name"),
                    "capital": country.get("capital"),
                    "currency_code": country.get("currency_code"),
                    "fips_code": country.get("fips_code"),
                    "country_iso2": country.get("country_iso2"),
                    "country_iso3": country.get("country_iso3"),
                    "continent": country.get("continent"),
                    "country_id": country.get("country_id"),
                    "currency_name": country.get("currency_name"),
                    "country_iso_numeric": country.get("country_iso_numeric"),
                    "phone_prefix": country.get("phone_prefix"),
                    "population": country.get("population"),
                }
            )
        return json.dumps(countries)
    except requests.RequestException as exc:
        return _error_response("fetching countries", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching countries", exc)


def random_cities_detailed_info(number_of_cities: int) -> str:
    """Get detailed info for random cities."""
    try:
        _validate_positive_int(number_of_cities, "number_of_cities")
        data = fetch_flight_data("cities", {"limit": number_of_cities})
        sampled_cities = _sample_data(data.get("data", []), number_of_cities)

        cities = []
        for city in sampled_cities:
            cities.append(
                {
                    "gmt": city.get("gmt"),
                    "city_id": city.get("city_id"),
                    "iata_code": city.get("iata_code"),
                    "country_iso2": city.get("country_iso2"),
                    "geoname_id": city.get("geoname_id"),
                    "latitude": city.get("latitude"),
                    "longitude": city.get("longitude"),
                    "timezone": city.get("timezone"),
                    "city_name": city.get("city_name"),
                }
            )
        return json.dumps(cities)
    except requests.RequestException as exc:
        return _error_response("fetching cities", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching cities", exc)


def list_airports(limit: int = 10, offset: int = 0, search: str = "") -> str:
    """List airports (supports basic-plan autocomplete through `search`)."""
    try:
        _validate_positive_int(limit, "limit")
        _validate_non_negative_int(offset, "offset")
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search

        return _list_reference_data(
            "airports",
            params,
            lambda airport: {
                "airport_name": airport.get("airport_name"),
                "iata_code": airport.get("iata_code"),
                "icao_code": airport.get("icao_code"),
                "city_iata_code": airport.get("city_iata_code"),
                "country_name": airport.get("country_name"),
                "country_iso2": airport.get("country_iso2"),
                "timezone": airport.get("timezone"),
                "gmt": airport.get("gmt"),
            },
        )
    except requests.RequestException as exc:
        return _error_response("fetching airports", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching airports", exc)


def list_airlines(limit: int = 10, offset: int = 0, search: str = "") -> str:
    """List airlines (supports basic-plan autocomplete through `search`)."""
    try:
        _validate_positive_int(limit, "limit")
        _validate_non_negative_int(offset, "offset")
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search

        return _list_reference_data(
            "airlines",
            params,
            lambda airline: {
                "airline_name": airline.get("airline_name"),
                "iata_code": airline.get("iata_code"),
                "icao_code": airline.get("icao_code"),
                "callsign": airline.get("callsign"),
                "status": airline.get("status"),
                "country_name": airline.get("country_name"),
                "country_iso2": airline.get("country_iso2"),
            },
        )
    except requests.RequestException as exc:
        return _error_response("fetching airlines", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching airlines", exc)


def list_routes(
    limit: int = 10,
    offset: int = 0,
    airline_iata: str = "",
    dep_iata: str = "",
    arr_iata: str = "",
) -> str:
    """List routes (available on Basic plan and higher)."""
    try:
        _validate_positive_int(limit, "limit")
        _validate_non_negative_int(offset, "offset")
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if airline_iata:
            params["airline_iata"] = airline_iata
        if dep_iata:
            params["dep_iata"] = dep_iata
        if arr_iata:
            params["arr_iata"] = arr_iata

        return _list_reference_data(
            "routes",
            params,
            lambda route: {
                "airline_iata": route.get("airline_iata"),
                "airline_icao": route.get("airline_icao"),
                "flight_number": route.get("flight_number"),
                "dep_iata": route.get("dep_iata"),
                "dep_icao": route.get("dep_icao"),
                "arr_iata": route.get("arr_iata"),
                "arr_icao": route.get("arr_icao"),
            },
        )
    except requests.RequestException as exc:
        return _error_response("fetching routes", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching routes", exc)


def list_taxes(limit: int = 10, offset: int = 0, search: str = "") -> str:
    """List aviation taxes (available on all plans)."""
    try:
        _validate_positive_int(limit, "limit")
        _validate_non_negative_int(offset, "offset")
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search

        return _list_reference_data(
            "taxes",
            params,
            lambda tax: {
                "tax_id": tax.get("tax_id"),
                "tax_name": tax.get("tax_name"),
                "iata_code": tax.get("iata_code"),
            },
        )
    except requests.RequestException as exc:
        return _error_response("fetching taxes", exc)
    except (KeyError, ValueError, TypeError) as exc:
        return _error_response("fetching taxes", exc)


@mcp.tool(
    name="flights_with_airline",
    description="Return a random sample of live flights filtered by airline name.",
)
def flights_with_airline_tool(
    airline_name: Annotated[
        str, Field(description="Airline name to filter flights (for example: Delta Air Lines).")
    ],
    number_of_flights: Annotated[int, Field(description="Number of random flights to return.", gt=0)],
) -> str:
    """Tool wrapper for flights_with_airline."""
    input = FlightsWithAirlineInput(
        airline_name=airline_name,
        number_of_flights=number_of_flights,
    )
    return flights_with_airline(
        airline_name=input.airline_name,
        number_of_flights=input.number_of_flights,
    )


@mcp.tool(
    name="historical_flights_by_date",
    description=(
        "Return a random sample of historical flights for a date with optional airline "
        "and route filters."
    ),
)
def historical_flights_by_date_tool(
    flight_date: Annotated[str, Field(description="Date in YYYY-MM-DD format.", examples=["2026-03-01"])],
    number_of_flights: Annotated[int, Field(description="Number of random flights to return.", gt=0)],
    airline_iata: Annotated[str, Field(description="Optional airline IATA code filter (for example: DL).")] = "",
    dep_iata: Annotated[
        str, Field(description="Optional departure airport IATA code filter (for example: JFK).")
    ] = "",
    arr_iata: Annotated[
        str, Field(description="Optional arrival airport IATA code filter (for example: LAX).")
    ] = "",
) -> str:
    """Tool wrapper for historical_flights_by_date."""
    input = HistoricalFlightsByDateInput(
        flight_date=flight_date,
        number_of_flights=number_of_flights,
        airline_iata=airline_iata,
        dep_iata=dep_iata,
        arr_iata=arr_iata,
    )
    return historical_flights_by_date(
        flight_date=input.flight_date,
        number_of_flights=input.number_of_flights,
        airline_iata=input.airline_iata,
        dep_iata=input.dep_iata,
        arr_iata=input.arr_iata,
    )


@mcp.tool(
    name="flight_arrival_departure_schedule",
    description=(
        "Return current-day arrival or departure schedule samples for an airport, "
        "optionally filtered by airline name."
    ),
)
def flight_arrival_departure_schedule_tool(
    airport_iata_code: Annotated[str, Field(description="Airport IATA code (for example: SFO).", min_length=1)],
    schedule_type: Annotated[str, Field(description="Schedule type: arrival or departure.")],
    airline_name: Annotated[str, Field(description="Optional airline name filter.")] = "",
    number_of_flights: Annotated[int, Field(description="Number of random flights to return.", gt=0)] = 5,
) -> str:
    """Tool wrapper for flight_arrival_departure_schedule."""
    input = FlightArrivalDepartureScheduleInput(
        airport_iata_code=airport_iata_code,
        schedule_type=schedule_type,
        airline_name=airline_name,
        number_of_flights=number_of_flights,
    )
    return flight_arrival_departure_schedule(
        airport_iata_code=input.airport_iata_code,
        schedule_type=input.schedule_type,
        airline_name=input.airline_name,
        number_of_flights=input.number_of_flights,
    )


@mcp.tool(
    name="future_flights_arrival_departure_schedule",
    description="Return future arrival or departure schedule samples for an airport and date.",
)
def future_flights_arrival_departure_schedule_tool(
    airport_iata_code: Annotated[str, Field(description="Airport IATA code (for example: SFO).", min_length=1)],
    schedule_type: Annotated[str, Field(description="Schedule type: arrival or departure.")],
    airline_iata: Annotated[str, Field(description="Optional airline IATA code filter (for example: UA).")] = "",
    date: Annotated[str, Field(description="Future date in YYYY-MM-DD format.", examples=["2026-03-01"])] = "",
    number_of_flights: Annotated[int, Field(description="Number of random flights to return.", gt=0)] = 5,
) -> str:
    """Tool wrapper for future_flights_arrival_departure_schedule."""
    input = FutureFlightsArrivalDepartureScheduleInput(
        airport_iata_code=airport_iata_code,
        schedule_type=schedule_type,
        airline_iata=airline_iata,
        date=date,
        number_of_flights=number_of_flights,
    )
    return future_flights_arrival_departure_schedule(
        airport_iata_code=input.airport_iata_code,
        schedule_type=input.schedule_type,
        airline_iata=input.airline_iata,
        date=input.date,
        number_of_flights=input.number_of_flights,
    )


@mcp.tool(
    name="random_aircraft_type",
    description="Return random aircraft type records.",
)
def random_aircraft_type_tool(
    number_of_aircraft: Annotated[
        int, Field(description="Number of random aircraft types to return.", gt=0)
    ],
) -> str:
    """Tool wrapper for random_aircraft_type."""
    input = RandomAircraftTypeInput(number_of_aircraft=number_of_aircraft)
    return random_aircraft_type(number_of_aircraft=input.number_of_aircraft)


@mcp.tool(
    name="random_airplanes_detailed_info",
    description="Return detailed metadata for random airplanes.",
)
def random_airplanes_detailed_info_tool(
    number_of_airplanes: Annotated[int, Field(description="Number of random airplanes to return.", gt=0)],
) -> str:
    """Tool wrapper for random_airplanes_detailed_info."""
    input = RandomAirplanesDetailedInfoInput(number_of_airplanes=number_of_airplanes)
    return random_airplanes_detailed_info(number_of_airplanes=input.number_of_airplanes)


@mcp.tool(
    name="random_countries_detailed_info",
    description="Return detailed metadata for random countries.",
)
def random_countries_detailed_info_tool(
    number_of_countries: Annotated[int, Field(description="Number of random countries to return.", gt=0)],
) -> str:
    """Tool wrapper for random_countries_detailed_info."""
    input = RandomCountriesDetailedInfoInput(number_of_countries=number_of_countries)
    return random_countries_detailed_info(number_of_countries=input.number_of_countries)


@mcp.tool(
    name="random_cities_detailed_info",
    description="Return detailed metadata for random cities.",
)
def random_cities_detailed_info_tool(
    number_of_cities: Annotated[int, Field(description="Number of random cities to return.", gt=0)],
) -> str:
    """Tool wrapper for random_cities_detailed_info."""
    input = RandomCitiesDetailedInfoInput(number_of_cities=number_of_cities)
    return random_cities_detailed_info(number_of_cities=input.number_of_cities)


@mcp.tool(
    name="list_airports",
    description="List airports with pagination and optional search.",
)
def list_airports_tool(
    limit: Annotated[int, Field(description="Maximum number of airports to return.", gt=0)] = 10,
    offset: Annotated[int, Field(description="Offset for pagination.", ge=0)] = 0,
    search: Annotated[str, Field(description="Optional airport search text for autocomplete.")] = "",
) -> str:
    """Tool wrapper for list_airports."""
    input = ListAirportsInput(limit=limit, offset=offset, search=search)
    return list_airports(limit=input.limit, offset=input.offset, search=input.search)


@mcp.tool(
    name="list_airlines",
    description="List airlines with pagination and optional search.",
)
def list_airlines_tool(
    limit: Annotated[int, Field(description="Maximum number of airlines to return.", gt=0)] = 10,
    offset: Annotated[int, Field(description="Offset for pagination.", ge=0)] = 0,
    search: Annotated[str, Field(description="Optional airline search text for autocomplete.")] = "",
) -> str:
    """Tool wrapper for list_airlines."""
    input = ListAirlinesInput(limit=limit, offset=offset, search=search)
    return list_airlines(limit=input.limit, offset=input.offset, search=input.search)


@mcp.tool(
    name="list_routes",
    description="List routes with pagination and optional airline/departure/arrival filters.",
)
def list_routes_tool(
    limit: Annotated[int, Field(description="Maximum number of routes to return.", gt=0)] = 10,
    offset: Annotated[int, Field(description="Offset for pagination.", ge=0)] = 0,
    airline_iata: Annotated[str, Field(description="Optional airline IATA code filter.")] = "",
    dep_iata: Annotated[str, Field(description="Optional departure airport IATA code filter.")] = "",
    arr_iata: Annotated[str, Field(description="Optional arrival airport IATA code filter.")] = "",
) -> str:
    """Tool wrapper for list_routes."""
    input = ListRoutesInput(
        limit=limit,
        offset=offset,
        airline_iata=airline_iata,
        dep_iata=dep_iata,
        arr_iata=arr_iata,
    )
    return list_routes(
        limit=input.limit,
        offset=input.offset,
        airline_iata=input.airline_iata,
        dep_iata=input.dep_iata,
        arr_iata=input.arr_iata,
    )


@mcp.tool(
    name="list_taxes",
    description="List aviation taxes with pagination and optional search.",
)
def list_taxes_tool(
    limit: Annotated[int, Field(description="Maximum number of tax records to return.", gt=0)] = 10,
    offset: Annotated[int, Field(description="Offset for pagination.", ge=0)] = 0,
    search: Annotated[str, Field(description="Optional tax search text.")] = "",
) -> str:
    """Tool wrapper for list_taxes."""
    input = ListTaxesInput(limit=limit, offset=offset, search=search)
    return list_taxes(limit=input.limit, offset=input.offset, search=input.search)


@mcp.prompt(
    name="plan_airline_flight_lookup",
    description="Generate a concise plan for querying live flights for an airline.",
)
def plan_airline_flight_lookup(
    airline_name: Annotated[str, Field(description="Airline name to query live flights for.")],
    number_of_flights: Annotated[int, Field(description="Number of flights to request.", gt=0)] = 5,
) -> str:
    """Prompt for guiding flight lookup."""
    return (
        f"Use the `flights_with_airline` tool with airline_name='{airline_name}' and "
        f"number_of_flights={number_of_flights}. Return a compact summary including "
        "flight number, departure airport, arrival airport, and current status."
    )


@mcp.prompt(
    name="plan_future_schedule_lookup",
    description="Generate a plan for querying future arrival or departure schedules.",
)
def plan_future_schedule_lookup(
    airport_iata_code: Annotated[str, Field(description="Airport IATA code (for example: SFO).")],
    date: Annotated[str, Field(description="Future date in YYYY-MM-DD format.", examples=["2026-03-01"])],
    schedule_type: Annotated[str, Field(description="Schedule type: arrival or departure.")] = "departure",
) -> str:
    """Prompt for future schedule lookup."""
    return (
        "Use the `future_flights_arrival_departure_schedule` tool with "
        f"airport_iata_code='{airport_iata_code}', date='{date}', "
        f"schedule_type='{schedule_type}', airline_iata='', and number_of_flights=5. "
        "Summarize carriers, schedule times, and aircraft types."
    )


@mcp.prompt(
    name="plan_reference_data_lookup",
    description="Generate a plan for exploring airport/airline/route/tax reference data.",
)
def plan_reference_data_lookup(
    data_type: Annotated[
        str, Field(description="Reference data category: airports, airlines, routes, or taxes.")
    ] = "airports",
    search: Annotated[str, Field(description="Optional search term to narrow results.")] = "",
) -> str:
    """Prompt for reference-data lookup."""
    mapping = {
        "airports": "list_airports",
        "airlines": "list_airlines",
        "routes": "list_routes",
        "taxes": "list_taxes",
    }
    tool_name = mapping.get(data_type.lower(), "list_airports")
    return (
        f"Use `{tool_name}` with limit=10, offset=0, search='{search}'. "
        "If results are empty, broaden the query and retry once with an empty search."
    )


@mcp.resource(
    "aviationstack://meta/server",
    name="server_metadata",
    description="Static metadata about this Aviationstack MCP server.",
    mime_type="application/json",
)
def server_metadata_resource() -> str:
    """Resource containing top-level server metadata."""
    return json.dumps(
        {
            "name": "Aviationstack MCP",
            "api_base_url": API_BASE_URL,
            "required_config_key": "aviationstack_api_key",
            "auth_env_fallbacks": [
                "AVIATION_STACK_API_KEY",
                "AVIATIONSTACK_API_KEY",
                "aviationstack_api_key",
            ],
        },
        indent=2,
    )


@mcp.resource(
    "aviationstack://meta/endpoints",
    name="aviationstack_endpoints",
    description="Available Aviationstack API endpoints used by tools.",
    mime_type="application/json",
)
def aviationstack_endpoints_resource() -> str:
    """Resource listing endpoints used by the server."""
    return json.dumps(
        {
            "flights": "/flights",
            "timetable": "/timetable",
            "future_flights": "/flightsFuture",
            "aircraft_types": "/aircraft_types",
            "airplanes": "/airplanes",
            "countries": "/countries",
            "cities": "/cities",
            "airports": "/airports",
            "airlines": "/airlines",
            "routes": "/routes",
            "taxes": "/taxes",
        },
        indent=2,
    )


@mcp.resource(
    "aviationstack://examples/tool-input/{tool_name}",
    name="tool_input_examples",
    description="Example input payloads for each tool name.",
    mime_type="application/json",
)
def tool_input_examples_resource(tool_name: str) -> str:
    """Resource returning sample payload for a tool."""
    samples = {
        "flights_with_airline": {
            "airline_name": "Delta Air Lines",
            "number_of_flights": 5,
        },
        "historical_flights_by_date": {
            "flight_date": "2026-03-01",
            "number_of_flights": 5,
            "airline_iata": "DL",
            "dep_iata": "JFK",
            "arr_iata": "LAX",
        },
        "list_airports": {"limit": 10, "offset": 0, "search": "San"},
        "list_airlines": {"limit": 10, "offset": 0, "search": "Delta"},
        "list_routes": {
            "limit": 10,
            "offset": 0,
            "airline_iata": "DL",
            "dep_iata": "JFK",
            "arr_iata": "LAX",
        },
        "list_taxes": {"limit": 10, "offset": 0, "search": "US"},
    }
    return json.dumps(samples.get(tool_name, {"error": f"Unknown tool '{tool_name}'"}), indent=2)
