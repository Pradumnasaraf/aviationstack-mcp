import json
import os
import unittest
from unittest.mock import patch

from aviationstack_mcp import server


class MockResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self._status_ok:
            raise server.requests.HTTPError("403 Client Error: Forbidden")


class ServerToolTests(unittest.TestCase):
    def setUp(self):
        os.environ["AVIATION_STACK_API_KEY"] = "test-key"

    def test_future_flights_invalid_date_returns_structured_error(self):
        output = server.future_flights_arrival_departure_schedule(
            airport_iata_code="JFK",
            schedule_type="departure",
            airline_iata="DL",
            date="2026/03/01",
            number_of_flights=1,
        )
        parsed = json.loads(output)
        self.assertFalse(parsed["ok"])
        self.assertEqual(parsed["context"], "fetching flight future schedule")
        self.assertIn("YYYY-MM-DD", parsed["error"])

    def test_historical_flights_invalid_date_returns_structured_error(self):
        output = server.historical_flights_by_date(
            flight_date="01-03-2026",
            number_of_flights=1,
            airline_iata="DL",
            dep_iata="JFK",
            arr_iata="LAX",
        )
        parsed = json.loads(output)
        self.assertFalse(parsed["ok"])
        self.assertEqual(parsed["context"], "fetching historical flights")
        self.assertIn("YYYY-MM-DD", parsed["error"])

    def test_fetch_flight_data_surfaces_api_error_body(self):
        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse(
                {
                    "error": {
                        "code": "function_access_restricted",
                        "type": "api_error",
                        "message": "Your current subscription plan does not support this API function.",
                    }
                },
                status_ok=False,
            ),
        ):
            with self.assertRaises(ValueError) as ctx:
                server.fetch_flight_data("routes", {"limit": 1})
        self.assertIn("function_access_restricted", str(ctx.exception))
        self.assertIn("subscription plan", str(ctx.exception))

    def test_list_routes_returns_structured_error_envelope(self):
        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse(
                {
                    "error": {
                        "code": "function_access_restricted",
                        "type": "api_error",
                        "message": "Your current subscription plan does not support this API function.",
                    }
                },
                status_ok=False,
            ),
        ):
            output = server.list_routes(limit=1, offset=0, airline_iata="DL")
        parsed = json.loads(output)
        self.assertFalse(parsed["ok"])
        self.assertEqual(parsed["context"], "fetching routes")
        self.assertIn("function_access_restricted", parsed["error"])

    def test_list_airports_uses_shared_mapper_path(self):
        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse(
                {
                    "data": [
                        {
                            "airport_name": "San Francisco International",
                            "iata_code": "SFO",
                            "icao_code": "KSFO",
                            "city_iata_code": "SFO",
                            "country_name": "United States",
                            "country_iso2": "US",
                            "timezone": "America/Los_Angeles",
                            "gmt": "-8",
                        }
                    ]
                }
            ),
        ):
            output = server.list_airports(limit=1, offset=0, search="San")
        parsed = json.loads(output)
        self.assertIsInstance(parsed, list)
        self.assertEqual(parsed[0]["iata_code"], "SFO")


if __name__ == "__main__":
    unittest.main()
