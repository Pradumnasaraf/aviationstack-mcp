"""Unit tests for MCP server helpers and tool wrappers."""
# pylint: disable=protected-access

import json
import os
import unittest
from unittest.mock import patch

from aviationstack_mcp import server

RESTRICTED_PLAN_MESSAGE = (
    "Your current subscription plan does not support this API function."
)

RESTRICTED_PLAN_PAYLOAD = {
    "error": {
        "code": "function_access_restricted",
        "type": "api_error",
        "message": RESTRICTED_PLAN_MESSAGE,
    }
}


class MockResponse:
    """Simple response stub for requests calls."""

    def __init__(self, payload, status_ok=True):
        """Store payload and whether raise_for_status should fail."""
        self._payload = payload
        self._status_ok = status_ok

    def json(self):
        """Return the provided response payload."""
        return self._payload

    def raise_for_status(self):
        """Mimic requests behavior by raising on non-OK responses."""
        if not self._status_ok:
            raise server.requests.HTTPError("403 Client Error: Forbidden")


class RecordingGet:
    """Stub for requests.get that records the params it was called with."""

    def __init__(self, payload):
        """Store the payload to return and prepare the call log."""
        self.payload = payload
        self.calls = []

    def __call__(self, url, params=None, timeout=None):
        """Record the request and return the configured payload."""
        self.calls.append({"url": url, "params": params or {}, "timeout": timeout})
        return MockResponse(self.payload)

    def last_params(self):
        """Return the params of the most recent recorded request."""
        return self.calls[-1]["params"]


def flight_payload():
    """Return a minimal /flights response body."""
    return {
        "pagination": {"limit": 1, "offset": 0, "count": 1, "total": 1},
        "data": [
            {
                "flight_date": "2026-03-01",
                "flight_status": "active",
                "flight": {"iata": "AA100"},
                "airline": {"name": "American Airlines"},
                "departure": {
                    "airport": "John F Kennedy International",
                    "iata": "JFK",
                    "scheduled": "2026-03-01T18:00:00+00:00",
                    "terminal": "8",
                    "gate": "12",
                    "delay": 5,
                },
                "arrival": {
                    "airport": "Heathrow",
                    "iata": "LHR",
                    "scheduled": "2026-03-02T06:00:00+00:00",
                    "terminal": "3",
                    "gate": "A1",
                    "baggage": "5",
                    "delay": None,
                },
            }
        ],
    }


class SecretScrubbingTests(unittest.TestCase):
    """The API key must never reach the client through an error message."""

    def setUp(self):
        """Set a deterministic API key for tests."""
        os.environ["AVIATION_STACK_API_KEY"] = "secret-key-value"

    def test_scrub_secrets_redacts_access_key_query_param(self):
        """The access_key query parameter should be redacted."""
        text = "url: /v1/airports?access_key=secret-key-value&limit=5"
        self.assertEqual(
            server._scrub_secrets(text), "url: /v1/airports?access_key=***&limit=5"
        )

    def test_scrub_secrets_redacts_bare_key_occurrence(self):
        """A bare key occurrence outside a query string should also be redacted."""
        self.assertNotIn(
            "secret-key-value", server._scrub_secrets("key was secret-key-value here")
        )

    def test_connection_error_does_not_leak_api_key(self):
        """requests puts the request URL in connection errors, so it must be scrubbed."""

        def raise_connection_error(url, params=None, timeout=None):
            del timeout
            prepared = server.requests.Request("GET", url, params=params).prepare()
            raise server.requests.exceptions.ConnectionError(
                f"Max retries exceeded with url: {prepared.url}"
            )

        with patch("aviationstack_mcp.server.requests.get", raise_connection_error):
            output = server.list_airports(limit=5)

        self.assertNotIn("secret-key-value", output)
        parsed = json.loads(output)
        self.assertFalse(parsed["ok"])
        self.assertIn("access_key=***", parsed["error"])


class ResponseEnvelopeTests(unittest.TestCase):
    """Every tool returns the same success and error envelope."""

    def setUp(self):
        """Set a deterministic API key for tests."""
        os.environ["AVIATION_STACK_API_KEY"] = "test-key"

    def test_success_envelope_shape(self):
        """Success responses carry ok, count and data."""
        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse(flight_payload()),
        ):
            parsed = json.loads(server.get_flight_status(flight_iata="AA100"))

        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["count"], 1)
        self.assertEqual(parsed["data"][0]["flight_number"], "AA100")
        self.assertEqual(parsed["data"][0]["arrival_baggage"], "5")

    def test_requested_flight_is_listed_before_its_codeshares(self):
        """The API returns codeshares alongside the flight, so the exact match leads."""
        payload = {
            "data": [
                {"flight": {"iata": "BA1511"}, "airline": {"name": "British Airways"}},
                {"flight": {"iata": "AA100"}, "airline": {"name": "American Airlines"}},
            ]
        }
        with patch(
            "aviationstack_mcp.server.requests.get", return_value=MockResponse(payload)
        ):
            parsed = json.loads(server.get_flight_status(flight_iata="AA100"))

        self.assertEqual(parsed["count"], 2)
        self.assertEqual(parsed["data"][0]["flight_number"], "AA100")

    def test_empty_result_is_a_success_with_message(self):
        """An empty result is still a success envelope, not a bare string."""
        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse({"data": []}),
        ):
            parsed = json.loads(server.get_flight_status(flight_iata="ZZ999"))

        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["count"], 0)
        self.assertEqual(parsed["data"], [])
        self.assertIn("ZZ999", parsed["message"])

    def test_list_tools_expose_pagination_totals(self):
        """Reference listings pass the API pagination block through to the caller."""
        payload = {
            "pagination": {"limit": 1, "offset": 0, "count": 1, "total": 6427},
            "data": [{"airport_name": "San Francisco International", "iata_code": "SFO"}],
        }
        with patch(
            "aviationstack_mcp.server.requests.get", return_value=MockResponse(payload)
        ):
            parsed = json.loads(server.list_airports(limit=1, offset=0, search="San"))

        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["data"][0]["iata_code"], "SFO")
        self.assertEqual(parsed["pagination"]["total"], 6427)

    def test_error_envelope_preserves_api_error_code(self):
        """API errors are mapped into the shared error envelope."""
        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse(RESTRICTED_PLAN_PAYLOAD, status_ok=False),
        ):
            parsed = json.loads(server.list_routes(limit=1, offset=0, airline_iata="DL"))

        self.assertFalse(parsed["ok"])
        self.assertEqual(parsed["context"], "fetching routes")
        self.assertIn("function_access_restricted", parsed["error"])


class ValidationTests(unittest.TestCase):
    """Input validation happens before any request is made."""

    def setUp(self):
        """Set a deterministic API key for tests."""
        os.environ["AVIATION_STACK_API_KEY"] = "test-key"

    def test_limit_above_max_is_rejected(self):
        """A limit beyond MAX_LIMIT is refused so one call cannot drain the quota."""
        parsed = json.loads(server.list_airports(limit=server.MAX_LIMIT + 1))
        self.assertFalse(parsed["ok"])
        self.assertIn(str(server.MAX_LIMIT), parsed["error"])

    def test_limit_at_max_is_accepted(self):
        """The boundary value itself is allowed."""
        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse({"data": []}),
        ):
            parsed = json.loads(server.list_airports(limit=server.MAX_LIMIT))
        self.assertTrue(parsed["ok"])

    def test_unknown_flight_status_is_rejected(self):
        """An unsupported flight_status value is refused with the valid choices."""
        parsed = json.loads(
            server.flights_with_airline(
                airline_name="Delta Air Lines",
                number_of_flights=1,
                flight_status="airborne",
            )
        )
        self.assertFalse(parsed["ok"])
        self.assertIn("scheduled", parsed["error"])

    def test_future_flights_invalid_date_returns_structured_error(self):
        """Invalid date format should return the shared error envelope."""
        parsed = json.loads(
            server.future_flights_arrival_departure_schedule(
                airport_iata_code="JFK",
                schedule_type="departure",
                airline_iata="DL",
                date="2026/03/01",
                number_of_flights=1,
            )
        )
        self.assertFalse(parsed["ok"])
        self.assertEqual(parsed["context"], "fetching flight future schedule")
        self.assertIn("YYYY-MM-DD", parsed["error"])

    def test_historical_flights_invalid_date_returns_structured_error(self):
        """Historical flights should validate date format consistently."""
        parsed = json.loads(
            server.historical_flights_by_date(
                flight_date="01-03-2026",
                number_of_flights=1,
                airline_iata="DL",
                dep_iata="JFK",
                arr_iata="LAX",
            )
        )
        self.assertFalse(parsed["ok"])
        self.assertEqual(parsed["context"], "fetching historical flights")
        self.assertIn("YYYY-MM-DD", parsed["error"])

    def test_invalid_schedule_type_is_rejected(self):
        """Schedule type is restricted to arrival or departure."""
        parsed = json.loads(
            server.flight_arrival_departure_schedule(
                airport_iata_code="SFO",
                schedule_type="sideways",
                airline_name="",
                number_of_flights=1,
            )
        )
        self.assertFalse(parsed["ok"])
        self.assertIn("arrival", parsed["error"])

    def test_fetch_flight_data_surfaces_api_error_body(self):
        """API error body should be preserved in raised exceptions."""
        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse(RESTRICTED_PLAN_PAYLOAD, status_ok=False),
        ):
            with self.assertRaises(ValueError) as ctx:
                server.fetch_flight_data("routes", {"limit": 1})

        self.assertIn("function_access_restricted", str(ctx.exception))
        self.assertIn(RESTRICTED_PLAN_MESSAGE, str(ctx.exception))


class RequestParameterTests(unittest.TestCase):
    """Tools must send the filters and limits they advertise."""

    def setUp(self):
        """Set a deterministic API key for tests."""
        os.environ["AVIATION_STACK_API_KEY"] = "test-key"

    def test_timetable_sends_limit_to_the_api(self):
        """The schedule tool bounds the response server-side instead of over-fetching."""
        recorder = RecordingGet({"data": []})
        with patch("aviationstack_mcp.server.requests.get", recorder):
            server.flight_arrival_departure_schedule(
                airport_iata_code="SFO",
                schedule_type="departure",
                airline_name="",
                number_of_flights=3,
            )
        self.assertEqual(recorder.last_params()["limit"], 3)
        self.assertEqual(recorder.last_params()["type"], "departure")

    def test_future_flights_sends_limit_to_the_api(self):
        """The future schedule tool also bounds the response server-side."""
        recorder = RecordingGet({"data": []})
        with patch("aviationstack_mcp.server.requests.get", recorder):
            server.future_flights_arrival_departure_schedule(
                airport_iata_code="SFO",
                schedule_type="arrival",
                airline_iata="UA",
                date="2026-03-01",
                number_of_flights=4,
            )
        self.assertEqual(recorder.last_params()["limit"], 4)
        self.assertEqual(recorder.last_params()["date"], "2026-03-01")

    def test_get_flight_status_normalises_flight_number(self):
        """Flight numbers are upper-cased and trimmed before being sent."""
        recorder = RecordingGet(flight_payload())
        with patch("aviationstack_mcp.server.requests.get", recorder):
            server.get_flight_status(flight_iata="  aa100 ")
        self.assertEqual(recorder.last_params()["flight_iata"], "AA100")
        self.assertNotIn("flight_date", recorder.last_params())

    def test_flight_status_filter_is_forwarded(self):
        """A valid flight_status filter reaches the API in lower case."""
        recorder = RecordingGet({"data": []})
        with patch("aviationstack_mcp.server.requests.get", recorder):
            server.flights_with_airline(
                airline_name="Delta Air Lines",
                number_of_flights=2,
                flight_status="ACTIVE",
            )
        self.assertEqual(recorder.last_params()["flight_status"], "active")

    def test_api_key_is_sent_as_access_key(self):
        """The key is passed as the access_key query parameter."""
        recorder = RecordingGet({"data": []})
        with patch("aviationstack_mcp.server.requests.get", recorder):
            server.list_taxes(limit=1)
        self.assertEqual(recorder.last_params()["access_key"], "test-key")


class RandomSamplingTests(unittest.TestCase):
    """The random_* tools must actually vary the records they return."""

    def setUp(self):
        """Reset the cached endpoint totals and set an API key."""
        os.environ["AVIATION_STACK_API_KEY"] = "test-key"
        server._ENDPOINT_TOTALS.clear()

    def tearDown(self):
        """Leave no cached totals behind for other tests."""
        server._ENDPOINT_TOTALS.clear()

    def test_first_call_reads_from_offset_zero_and_caches_total(self):
        """Without a known total the first call starts at offset 0 and learns it."""
        recorder = RecordingGet(
            {
                "pagination": {"limit": 2, "offset": 0, "count": 2, "total": 312},
                "data": [{"aircraft_name": "A320"}, {"aircraft_name": "B738"}],
            }
        )
        with patch("aviationstack_mcp.server.requests.get", recorder):
            parsed = json.loads(server.random_aircraft_type(number_of_aircraft=2))

        self.assertEqual(recorder.last_params()["offset"], 0)
        self.assertEqual(server._ENDPOINT_TOTALS["aircraft_types"], 312)
        self.assertEqual(parsed["count"], 2)

    def test_later_calls_use_a_random_offset_within_the_total(self):
        """Once the total is known the offset is randomised inside the dataset."""
        server._ENDPOINT_TOTALS["cities"] = 9000
        recorder = RecordingGet({"data": [{"city_name": "Paris"}]})
        with patch("aviationstack_mcp.server.requests.get", recorder):
            with patch("aviationstack_mcp.server.random.randint", return_value=4321):
                server.random_cities_detailed_info(number_of_cities=1)

        self.assertEqual(recorder.last_params()["offset"], 4321)

    def test_random_offset_never_exceeds_the_last_page(self):
        """The offset is bounded so the API always has `count` records left to return."""
        server._ENDPOINT_TOTALS["countries"] = 250
        captured = {}

        def fake_randint(low, high):
            captured["low"] = low
            captured["high"] = high
            return high

        recorder = RecordingGet({"data": []})
        with patch("aviationstack_mcp.server.requests.get", recorder):
            with patch("aviationstack_mcp.server.random.randint", fake_randint):
                server.random_countries_detailed_info(number_of_countries=10)

        self.assertEqual(captured["low"], 0)
        self.assertEqual(captured["high"], 240)

    def test_single_request_per_call(self):
        """Randomising the offset must not cost an extra API request."""
        server._ENDPOINT_TOTALS["airplanes"] = 19000
        recorder = RecordingGet({"data": []})
        with patch("aviationstack_mcp.server.requests.get", recorder):
            server.random_airplanes_detailed_info(number_of_airplanes=5)
        self.assertEqual(len(recorder.calls), 1)


class ReferenceListingTests(unittest.TestCase):
    """The list_* tools map API payloads onto their documented fields."""

    def setUp(self):
        """Set a deterministic API key for tests."""
        os.environ["AVIATION_STACK_API_KEY"] = "test-key"

    def test_country_name_uses_the_field_the_api_returns(self):
        """The countries endpoint returns country_name, not name."""
        payload = {"data": [{"country_name": "Andorra", "capital": "Andorra la Vella"}]}
        with patch(
            "aviationstack_mcp.server.requests.get", return_value=MockResponse(payload)
        ):
            parsed = json.loads(server.random_countries_detailed_info(1))
        self.assertEqual(parsed["data"][0]["country_name"], "Andorra")

    def test_list_airlines_maps_records(self):
        """Airline records are reduced to the documented fields."""
        payload = {"data": [{"airline_name": "Delta Air Lines", "iata_code": "DL"}]}
        with patch(
            "aviationstack_mcp.server.requests.get", return_value=MockResponse(payload)
        ):
            parsed = json.loads(server.list_airlines(limit=1, search="Delta"))
        self.assertEqual(parsed["data"][0]["airline_name"], "Delta Air Lines")

    def test_list_routes_maps_records(self):
        """Route records are reduced to the documented fields."""
        payload = {"data": [{"airline_iata": "DL", "dep_iata": "JFK", "arr_iata": "LAX"}]}
        with patch(
            "aviationstack_mcp.server.requests.get", return_value=MockResponse(payload)
        ):
            parsed = json.loads(server.list_routes(limit=1, airline_iata="DL"))
        self.assertEqual(parsed["data"][0]["arr_iata"], "LAX")

    def test_list_taxes_maps_records(self):
        """Tax records are reduced to the documented fields."""
        payload = {"data": [{"tax_id": "1", "tax_name": "Passenger Fee", "iata_code": "US"}]}
        with patch(
            "aviationstack_mcp.server.requests.get", return_value=MockResponse(payload)
        ):
            parsed = json.loads(server.list_taxes(limit=1, search="US"))
        self.assertEqual(parsed["data"][0]["tax_name"], "Passenger Fee")

    def test_flights_with_airline_maps_records(self):
        """Live flight records are reduced to the documented fields."""
        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse(flight_payload()),
        ):
            parsed = json.loads(
                server.flights_with_airline(
                    airline_name="American Airlines", number_of_flights=1
                )
            )
        self.assertEqual(parsed["data"][0]["departure_gate"], "12")

    def test_historical_flights_maps_records(self):
        """Historical flight records are reduced to the documented fields."""
        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse(flight_payload()),
        ):
            parsed = json.loads(
                server.historical_flights_by_date(
                    flight_date="2026-03-01", number_of_flights=1
                )
            )
        self.assertEqual(parsed["data"][0]["flight_date"], "2026-03-01")


class ServerConstructionTests(unittest.TestCase):
    """Server creation degrades gracefully on older FastMCP releases."""

    def test_create_mcp_server_falls_back_when_config_schema_is_unsupported(self):
        """Server creation should retry without config_schema on older FastMCP versions."""
        call_kwargs = []

        def fake_fastmcp(_name, **kwargs):
            call_kwargs.append(kwargs.copy())
            if "config_schema" in kwargs:
                raise TypeError(
                    "FastMCP.__init__() got an unexpected keyword argument "
                    "'config_schema'"
                )
            return object()

        with patch("aviationstack_mcp.server.FastMCP", side_effect=fake_fastmcp):
            result = server.create_mcp_server()

        self.assertIsNotNone(result)
        self.assertEqual(len(call_kwargs), 2)
        self.assertIn("config_schema", call_kwargs[0])
        self.assertNotIn("config_schema", call_kwargs[1])


class MissingApiKeyTests(unittest.TestCase):
    """A missing key is reported clearly instead of hitting the API."""

    def test_missing_api_key_returns_error_envelope(self):
        """All key environment variables unset should produce a structured error."""
        for name in (
            "AVIATION_STACK_API_KEY",
            "AVIATIONSTACK_API_KEY",
            "aviationstack_api_key",
        ):
            os.environ.pop(name, None)

        parsed = json.loads(server.list_airports(limit=1))
        self.assertFalse(parsed["ok"])
        self.assertIn("API key not set", parsed["error"])
        os.environ["AVIATION_STACK_API_KEY"] = "test-key"


class ToolWrapperTests(unittest.TestCase):
    """Each registered tool wrapper validates input and delegates to its function."""

    def setUp(self):
        """Set a deterministic API key for tests."""
        os.environ["AVIATION_STACK_API_KEY"] = "test-key"
        server._ENDPOINT_TOTALS.clear()

    def tearDown(self):
        """Leave no cached totals behind for other tests."""
        server._ENDPOINT_TOTALS.clear()

    def test_every_wrapper_returns_the_shared_envelope(self):
        """All 13 wrappers round-trip through validation and return valid JSON."""
        calls = [
            (server.get_flight_status_tool, {"flight_iata": "AA100"}),
            (
                server.flights_with_airline_tool,
                {"airline_name": "Delta Air Lines", "number_of_flights": 1},
            ),
            (
                server.historical_flights_by_date_tool,
                {"flight_date": "2026-03-01", "number_of_flights": 1},
            ),
            (
                server.flight_arrival_departure_schedule_tool,
                {"airport_iata_code": "SFO", "schedule_type": "departure"},
            ),
            (
                server.future_flights_arrival_departure_schedule_tool,
                {
                    "airport_iata_code": "SFO",
                    "schedule_type": "arrival",
                    "date": "2026-03-01",
                },
            ),
            (server.random_aircraft_type_tool, {"number_of_aircraft": 1}),
            (server.random_airplanes_detailed_info_tool, {"number_of_airplanes": 1}),
            (server.random_countries_detailed_info_tool, {"number_of_countries": 1}),
            (server.random_cities_detailed_info_tool, {"number_of_cities": 1}),
            (server.list_airports_tool, {"limit": 1}),
            (server.list_airlines_tool, {"limit": 1}),
            (server.list_routes_tool, {"limit": 1}),
            (server.list_taxes_tool, {"limit": 1}),
        ]
        self.assertEqual(len(calls), 13)

        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse({"data": [], "pagination": {"total": 10}}),
        ):
            for wrapper, kwargs in calls:
                with self.subTest(tool=wrapper.__name__):
                    parsed = json.loads(wrapper(**kwargs))
                    self.assertTrue(parsed["ok"], parsed)
                    self.assertEqual(parsed["count"], 0)

    def test_wrapper_rejects_out_of_range_limit(self):
        """Pydantic validation on the wrapper rejects a limit above MAX_LIMIT."""
        with self.assertRaises(Exception):
            server.list_airports_tool(limit=server.MAX_LIMIT + 1)


class PromptTests(unittest.TestCase):
    """Prompts name the tool they are steering the model toward."""

    def test_flight_status_prompt_names_its_tool(self):
        """The single-flight prompt points at get_flight_status."""
        text = server.plan_flight_status_lookup(flight_iata="AA100")
        self.assertIn("get_flight_status", text)
        self.assertIn("AA100", text)
        self.assertIn("codeshare", text)

    def test_airline_prompt_names_its_tool(self):
        """The airline prompt points at flights_with_airline."""
        text = server.plan_airline_flight_lookup(airline_name="Delta Air Lines")
        self.assertIn("flights_with_airline", text)

    def test_future_schedule_prompt_names_its_tool(self):
        """The future schedule prompt points at its tool."""
        text = server.plan_future_schedule_lookup(
            airport_iata_code="SFO", date="2026-03-01"
        )
        self.assertIn("future_flights_arrival_departure_schedule", text)

    def test_reference_prompt_maps_each_category(self):
        """Each reference category resolves to its matching list tool."""
        for category, tool in [
            ("airports", "list_airports"),
            ("airlines", "list_airlines"),
            ("routes", "list_routes"),
            ("taxes", "list_taxes"),
        ]:
            with self.subTest(category=category):
                self.assertIn(tool, server.plan_reference_data_lookup(data_type=category))

    def test_reference_prompt_falls_back_for_unknown_category(self):
        """An unrecognised category falls back to airports rather than failing."""
        self.assertIn("list_airports", server.plan_reference_data_lookup(data_type="moons"))


class ResourceTests(unittest.TestCase):
    """Resources expose accurate, parseable metadata."""

    def test_server_metadata_lists_the_key_fallbacks(self):
        """The metadata resource documents every accepted key variable."""
        meta = json.loads(server.server_metadata_resource())
        self.assertEqual(meta["api_base_url"], server.API_BASE_URL)
        self.assertIn("AVIATION_STACK_API_KEY", meta["auth_env_fallbacks"])

    def test_endpoints_resource_covers_every_endpoint_the_tools_call(self):
        """The endpoints resource stays in step with the endpoints in use."""
        endpoints = json.loads(server.aviationstack_endpoints_resource())
        listed = {value.lstrip("/") for value in endpoints.values()}
        for endpoint in (
            "flights",
            "timetable",
            "flightsFuture",
            "aircraft_types",
            "airplanes",
            "countries",
            "cities",
            "airports",
            "airlines",
            "routes",
            "taxes",
        ):
            with self.subTest(endpoint=endpoint):
                self.assertIn(endpoint, listed)

    def test_tool_input_examples_cover_documented_tools(self):
        """Example payloads parse and include the newest tool."""
        example = json.loads(server.tool_input_examples_resource("get_flight_status"))
        self.assertEqual(example["flight_iata"], "AA100")

    def test_tool_input_examples_reports_unknown_tool(self):
        """An unknown tool name returns an explicit error rather than an empty object."""
        example = json.loads(server.tool_input_examples_resource("nope"))
        self.assertIn("nope", example["error"])


class NetworkFailureTests(unittest.TestCase):
    """Every tool converts a transport failure into the shared error envelope."""

    def setUp(self):
        """Set a deterministic API key for tests."""
        os.environ["AVIATION_STACK_API_KEY"] = "test-key"
        server._ENDPOINT_TOTALS.clear()

    def tearDown(self):
        """Leave no cached totals behind for other tests."""
        server._ENDPOINT_TOTALS.clear()

    def test_every_tool_handles_a_transport_error(self):
        """A RequestException never escapes as an exception to the caller."""
        calls = [
            ("fetching flight status", lambda: server.get_flight_status("AA100")),
            ("fetching flights", lambda: server.flights_with_airline("Delta", 1)),
            (
                "fetching historical flights",
                lambda: server.historical_flights_by_date("2026-03-01", 1),
            ),
            (
                "fetching flight schedule",
                lambda: server.flight_arrival_departure_schedule("SFO", "departure", "", 1),
            ),
            (
                "fetching flight future schedule",
                lambda: server.future_flights_arrival_departure_schedule(
                    "SFO", "arrival", "", "2026-03-01", 1
                ),
            ),
            ("fetching aircraft type", lambda: server.random_aircraft_type(1)),
            ("fetching airplanes", lambda: server.random_airplanes_detailed_info(1)),
            ("fetching countries", lambda: server.random_countries_detailed_info(1)),
            ("fetching cities", lambda: server.random_cities_detailed_info(1)),
            ("fetching airports", lambda: server.list_airports(1)),
            ("fetching airlines", lambda: server.list_airlines(1)),
            ("fetching routes", lambda: server.list_routes(1)),
            ("fetching taxes", lambda: server.list_taxes(1)),
        ]
        self.assertEqual(len(calls), 13)

        def boom(url, params=None, timeout=None):
            del url, params, timeout
            raise server.requests.exceptions.ConnectionError("connection refused")

        with patch("aviationstack_mcp.server.requests.get", boom):
            for context, call in calls:
                with self.subTest(context=context):
                    parsed = json.loads(call())
                    self.assertFalse(parsed["ok"])
                    self.assertEqual(parsed["context"], context)

    def test_every_tool_handles_a_malformed_record(self):
        """A record of the wrong shape becomes an error envelope, not a traceback."""
        calls = [
            lambda: server.get_flight_status("AA100"),
            lambda: server.flights_with_airline("Delta", 1),
            lambda: server.random_countries_detailed_info(1),
            lambda: server.list_airports(1),
        ]
        with patch(
            "aviationstack_mcp.server.requests.get",
            return_value=MockResponse({"data": ["not-a-record"]}),
        ):
            for call in calls:
                parsed = json.loads(call())
                self.assertFalse(parsed["ok"])

    def test_non_json_body_falls_back_to_status_check(self):
        """A body that is not JSON defers to raise_for_status."""

        class BadBody:
            """Response stub whose body cannot be decoded."""

            def json(self):
                """Raise the way requests does on a non-JSON body."""
                raise ValueError("no json")

            def raise_for_status(self):
                """Report the underlying HTTP failure."""
                raise server.requests.HTTPError("502 Bad Gateway")

        with patch("aviationstack_mcp.server.requests.get", return_value=BadBody()):
            parsed = json.loads(server.list_taxes(1))
        self.assertFalse(parsed["ok"])
        self.assertIn("502", parsed["error"])


class OptionalFilterTests(unittest.TestCase):
    """Optional filters are only sent when the caller supplies them."""

    def setUp(self):
        """Set a deterministic API key for tests."""
        os.environ["AVIATION_STACK_API_KEY"] = "test-key"

    def test_blank_flight_iata_is_rejected(self):
        """Whitespace is not a flight number."""
        parsed = json.loads(server.get_flight_status("   "))
        self.assertFalse(parsed["ok"])
        self.assertIn("must not be empty", parsed["error"])

    def test_flight_date_is_forwarded_when_given(self):
        """Supplying a date narrows the lookup to that day."""
        recorder = RecordingGet(flight_payload())
        with patch("aviationstack_mcp.server.requests.get", recorder):
            server.get_flight_status("AA100", flight_date="2026-03-01")
        self.assertEqual(recorder.last_params()["flight_date"], "2026-03-01")

    def test_historical_filters_are_forwarded(self):
        """Airline and route filters reach the API when set."""
        recorder = RecordingGet({"data": []})
        with patch("aviationstack_mcp.server.requests.get", recorder):
            server.historical_flights_by_date(
                "2026-03-01", 1, airline_iata="DL", dep_iata="JFK", arr_iata="LAX"
            )
        params = recorder.last_params()
        self.assertEqual(params["airline_iata"], "DL")
        self.assertEqual(params["dep_iata"], "JFK")
        self.assertEqual(params["arr_iata"], "LAX")

    def test_route_filters_are_forwarded(self):
        """Route filters reach the API when set."""
        recorder = RecordingGet({"data": []})
        with patch("aviationstack_mcp.server.requests.get", recorder):
            server.list_routes(1, 0, "DL", "JFK", "LAX")
        params = recorder.last_params()
        self.assertEqual(params["dep_iata"], "JFK")
        self.assertEqual(params["arr_iata"], "LAX")

    def test_airline_name_filter_is_forwarded_to_timetable(self):
        """The schedule tool forwards an airline name filter."""
        recorder = RecordingGet({"data": []})
        with patch("aviationstack_mcp.server.requests.get", recorder):
            server.flight_arrival_departure_schedule("SFO", "departure", "Delta", 1)
        self.assertEqual(recorder.last_params()["airline_name"], "Delta")

    def test_omitted_filters_are_not_sent(self):
        """Blank optional filters are left out of the request entirely."""
        recorder = RecordingGet({"data": []})
        with patch("aviationstack_mcp.server.requests.get", recorder):
            server.list_routes(1)
        for absent in ("airline_iata", "dep_iata", "arr_iata"):
            self.assertNotIn(absent, recorder.last_params())


class ValidationHelperTests(unittest.TestCase):
    """The shared validators reject bad values with actionable messages."""

    def test_positive_int_rejects_zero(self):
        """Zero is not a positive count."""
        with self.assertRaises(ValueError):
            server._validate_positive_int(0, "count")

    def test_non_negative_int_rejects_negative(self):
        """Offsets cannot be negative."""
        with self.assertRaises(ValueError):
            server._validate_non_negative_int(-1, "offset")

    def test_iso_date_rejects_impossible_day(self):
        """A calendar-invalid date is refused."""
        with self.assertRaises(ValueError):
            server._validate_iso_date("2026-02-31", "flight_date")

    def test_invalid_schedule_type_message_names_both_options(self):
        """The schedule type error tells the caller what is allowed."""
        parsed = json.loads(
            server.future_flights_arrival_departure_schedule(
                "SFO", "sideways", "", "2026-03-01", 1
            )
        )
        self.assertIn("departure", parsed["error"])


if __name__ == "__main__":
    unittest.main()
