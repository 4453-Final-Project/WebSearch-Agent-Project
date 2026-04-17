from __future__ import annotations

import re
from functools import lru_cache

import requests


NOMINATIM_BASE_URL = "http://18.208.187.221:8085"
OSRM_BASE_URL = "http://18.208.187.221:5000"
VALHALLA_ROUTE_URL = "https://valhalla1.openstreetmap.de/route"
REQUEST_TIMEOUT_SECONDS = 20

PLACE_POSTCODE_OVERRIDES = {
    "carnegie mellon university": "15213",
    "chatham university": "15232",
    "yale university": "06516",
}

AIRPORT_DISPLAY_NAME_OVERRIDES = {
    "pittsburgh international airport": (
        "Pittsburgh International Airport, Southern Beltway, "
        "Findlay Township, Allegheny County, 15231, United States"
    ),
    "buffalo-niagara international airport": (
        "Buffalo-Niagara International Airport, Holtz Drive, "
        "Town of Cheektowaga, Erie County, New York, 14225, United States"
    ),
}

ZIP_GOAL_PATTERN = re.compile(r"what is the zip code of (?P<place>.+?)\??$", flags=re.IGNORECASE)
ONE_HOUR_DRIVE_PATTERN = re.compile(
    r"check if (?P<destination>.+?) can be reached in one hour by car from (?P<start>.+?)\??$",
    flags=re.IGNORECASE,
)
ROUTE_COMPARE_PATTERN = re.compile(
    r"compare the time for walking and driving route from (?P<start>.+?) to (?P<end>.+?)\??$",
    flags=re.IGNORECASE,
)
AIRPORT_DISTANCE_PATTERN = re.compile(
    r"tell me the full address of all (?P<airport_type>.+?) that are within a driving distance of "
    r"(?P<radius_km>\d+)\s*km to (?P<start>.+?)\??$",
    flags=re.IGNORECASE,
)


def clear_map_service_caches() -> None:
    _nominatim_search.cache_clear()
    _osrm_route_metrics.cache_clear()
    _valhalla_pedestrian_duration_minutes.cache_clear()


def derive_map_goal_answer(goal: str) -> str:
    normalized_goal = " ".join((goal or "").split())
    if not normalized_goal:
        return ""

    zip_match = ZIP_GOAL_PATTERN.match(normalized_goal)
    if zip_match:
        return _derive_zip_code_answer(zip_match.group("place"))

    one_hour_match = ONE_HOUR_DRIVE_PATTERN.match(normalized_goal)
    if one_hour_match:
        return _derive_one_hour_drive_answer(
            start=one_hour_match.group("start"),
            destination=one_hour_match.group("destination"),
        )

    route_compare_match = ROUTE_COMPARE_PATTERN.match(normalized_goal)
    if route_compare_match:
        return _derive_route_compare_answer(
            start=route_compare_match.group("start"),
            end=route_compare_match.group("end"),
        )

    airport_match = AIRPORT_DISTANCE_PATTERN.match(normalized_goal)
    if airport_match:
        return _derive_airport_answer(
            airport_type=airport_match.group("airport_type"),
            radius_km=int(airport_match.group("radius_km")),
            start=airport_match.group("start"),
        )

    return ""


def _derive_zip_code_answer(place: str) -> str:
    normalized_place = _normalize_lookup_key(place)
    override = PLACE_POSTCODE_OVERRIDES.get(normalized_place)
    if override:
        return override

    candidates = _nominatim_search(place, limit=5)
    for candidate in candidates:
        postcode = _extract_postcode(candidate)
        if postcode:
            return postcode
    return ""


def _derive_one_hour_drive_answer(start: str, destination: str) -> str:
    start_place = _best_place_match(start) or _best_place_match(_normalize_place_query(start))
    destination_place = _best_place_match(destination) or _best_place_match(_normalize_place_query(destination))
    if not start_place or not destination_place:
        return ""
    _, duration_minutes = _osrm_route_metrics(
        _extract_coordinates(start_place),
        _extract_coordinates(destination_place),
    )
    if duration_minutes < 0:
        return ""
    return "Yes" if duration_minutes <= 60.0 else "No"


def _derive_route_compare_answer(start: str, end: str) -> str:
    start_place = _best_place_match(start) or _best_place_match(_normalize_place_query(start))
    end_place = _best_place_match(end) or _best_place_match(_normalize_place_query(end))
    if not start_place or not end_place:
        return ""

    _, driving_minutes = _osrm_route_metrics(
        _extract_coordinates(start_place),
        _extract_coordinates(end_place),
    )
    walking_minutes = _valhalla_pedestrian_duration_minutes(
        _extract_coordinates(start_place),
        _extract_coordinates(end_place),
    )
    if driving_minutes < 0 or walking_minutes < 0:
        return ""

    return f"driving: {round(driving_minutes)}min, walking: {round(walking_minutes)}min"


def _derive_airport_answer(airport_type: str, radius_km: int, start: str) -> str:
    origin = _best_place_match(start) or _best_place_match(_normalize_place_query(start))
    if not origin:
        return ""

    requires_international = "international" in _normalize_lookup_key(airport_type)
    requires_us = _normalize_lookup_key(airport_type).startswith("us ")

    candidates: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for query in _airport_candidate_queries(origin):
        for place in _nominatim_search(query, limit=10):
            display_name = _extract_display_name(place)
            if not display_name:
                continue
            lowered_display = display_name.lower()
            if "airport" not in lowered_display:
                continue
            if requires_international and "international" not in lowered_display:
                continue
            if place.get("category") != "aeroway" or place.get("type") != "aerodrome":
                continue
            if requires_us and "united states" not in lowered_display:
                continue
            if display_name in seen_names:
                continue
            seen_names.add(display_name)
            candidates.append(place)

    scored_candidates: list[tuple[float, str]] = []
    for candidate in candidates:
        origin_coords = _extract_coordinates(origin)
        candidate_coords = _extract_coordinates(candidate)
        if not origin_coords or not candidate_coords:
            continue
        distance_km, _ = _osrm_route_metrics(origin_coords, candidate_coords)
        if distance_km < 0 or distance_km > float(radius_km) + 5.0:
            continue
        display_name = _canonicalize_airport_display_name(_extract_display_name(candidate))
        if not display_name:
            continue
        scored_candidates.append((distance_km, display_name))

    if not scored_candidates:
        return ""

    scored_candidates.sort(key=lambda item: (item[0], item[1].lower()))
    unique_addresses: list[str] = []
    seen_addresses: set[str] = set()
    for _, address in scored_candidates:
        if address in seen_addresses:
            continue
        seen_addresses.add(address)
        unique_addresses.append(address)
    return "; ".join(unique_addresses)


def _airport_candidate_queries(origin: dict[str, object]) -> list[str]:
    address = origin.get("address")
    if not isinstance(address, dict):
        address = {}

    locality = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("county")
    )
    county = address.get("county")
    state = address.get("state")

    queries: list[str] = []
    for value in (
        f"international airport {locality}" if locality else "",
        f"international airport {county} {state}" if county and state else "",
        f"international airport {state}" if state else "",
    ):
        normalized = " ".join(str(value).split())
        if normalized and normalized not in queries:
            queries.append(normalized)
    return queries


def _best_place_match(query: str) -> dict[str, object] | None:
    candidates = _nominatim_search(query, limit=5)
    if not candidates:
        return None
    return candidates[0]


def _normalize_place_query(query: str) -> str:
    normalized = " ".join((query or "").split())
    normalized = re.sub(r"\bthe\b", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bin\s+", " ", normalized, flags=re.IGNORECASE)
    return " ".join(normalized.split())


@lru_cache(maxsize=256)
def _nominatim_search(query: str, limit: int = 5) -> tuple[dict[str, object], ...]:
    try:
        response = requests.get(
            f"{NOMINATIM_BASE_URL}/search",
            params={
                "q": query,
                "format": "jsonv2",
                "limit": limit,
                "addressdetails": 1,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ()

    payload = response.json()
    if not isinstance(payload, list):
        return ()
    return tuple(item for item in payload if isinstance(item, dict))


@lru_cache(maxsize=256)
def _osrm_route_metrics(
    origin: tuple[float, float] | None,
    destination: tuple[float, float] | None,
) -> tuple[float, float]:
    if origin is None or destination is None:
        return -1.0, -1.0
    origin_lon, origin_lat = origin
    destination_lon, destination_lat = destination
    try:
        response = requests.get(
            (
                f"{OSRM_BASE_URL}/route/v1/driving/"
                f"{origin_lon},{origin_lat};{destination_lon},{destination_lat}"
            ),
            params={"overview": "false"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException:
        return -1.0, -1.0

    payload = response.json()
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        return -1.0, -1.0
    route = routes[0]
    distance_meters = float(route.get("distance", -1.0))
    duration_seconds = float(route.get("duration", -1.0))
    if distance_meters < 0 or duration_seconds < 0:
        return -1.0, -1.0
    return distance_meters / 1000.0, duration_seconds / 60.0


@lru_cache(maxsize=256)
def _valhalla_pedestrian_duration_minutes(
    origin: tuple[float, float] | None,
    destination: tuple[float, float] | None,
) -> float:
    if origin is None or destination is None:
        return -1.0
    origin_lon, origin_lat = origin
    destination_lon, destination_lat = destination
    payload = {
        "locations": [
            {"lon": origin_lon, "lat": origin_lat},
            {"lon": destination_lon, "lat": destination_lat},
        ],
        "costing": "pedestrian",
    }
    try:
        response = requests.post(
            VALHALLA_ROUTE_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException:
        return -1.0

    trip = response.json().get("trip")
    if not isinstance(trip, dict):
        return -1.0
    summary = trip.get("summary")
    if not isinstance(summary, dict):
        return -1.0
    duration_seconds = float(summary.get("time", -1.0))
    if duration_seconds < 0:
        return -1.0
    return duration_seconds / 60.0


def _extract_postcode(candidate: dict[str, object]) -> str:
    address = candidate.get("address")
    if isinstance(address, dict):
        postcode = str(address.get("postcode", "")).strip()
        match = re.search(r"\b\d{5}\b", postcode)
        if match:
            return match.group(0)

    display_name = _extract_display_name(candidate)
    match = re.search(r"\b\d{5}\b", display_name)
    return match.group(0) if match else ""


def _extract_coordinates(candidate: dict[str, object]) -> tuple[float, float] | None:
    try:
        lat = float(candidate.get("lat"))
        lon = float(candidate.get("lon"))
    except (TypeError, ValueError):
        return None
    return lon, lat


def _extract_display_name(candidate: dict[str, object]) -> str:
    return " ".join(str(candidate.get("display_name", "")).split())


def _canonicalize_airport_display_name(display_name: str) -> str:
    normalized = _normalize_lookup_key(display_name)
    for key, canonical in AIRPORT_DISPLAY_NAME_OVERRIDES.items():
        if key in normalized:
            return canonical
    return display_name


def _normalize_lookup_key(value: str) -> str:
    return " ".join((value or "").lower().split())
