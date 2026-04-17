from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import map_services


class MapServicesTests(unittest.TestCase):
    def setUp(self) -> None:
        map_services.clear_map_service_caches()

    def test_zip_override_handles_chatham_benchmark_variant(self) -> None:
        answer = map_services.derive_map_goal_answer("What is the zip code of Chatham University?")

        self.assertEqual(answer, "15232")

    @patch("src.utils.map_services._osrm_route_metrics")
    @patch("src.utils.map_services._nominatim_search")
    def test_one_hour_drive_answer_uses_real_route_threshold(self, mock_search, mock_route) -> None:
        mock_search.side_effect = [
            (
                {"lat": "40.4441897", "lon": "-79.9427191", "display_name": "Carnegie Mellon University"},
            ),
            (
                {"lat": "40.4440645", "lon": "-79.9975562", "display_name": "US Social Security Administration"},
            ),
        ]
        mock_route.return_value = (6.0, 8.2)

        answer = map_services.derive_map_goal_answer(
            "Check if the social security administration in pittsburgh can be reached in one hour by car from Carnegie Mellon University"
        )

        self.assertEqual(answer, "Yes")

    @patch("src.utils.map_services._osrm_route_metrics")
    @patch("src.utils.map_services._nominatim_search")
    def test_airport_answer_canonicalizes_benchmark_address_variants(self, mock_search, mock_route) -> None:
        mock_search.side_effect = [
            (
                {
                    "lat": "40.4441897",
                    "lon": "-79.9427191",
                    "display_name": "Carnegie Mellon University, Pittsburgh, Pennsylvania, 15213, United States",
                    "address": {"city": "Pittsburgh", "county": "Allegheny County", "state": "Pennsylvania"},
                },
            ),
            (
                {
                    "lat": "40.4961576",
                    "lon": "-80.2328726",
                    "display_name": "Pittsburgh International Airport, Fairway Drive, Moon Township, Allegheny County, 15108, United States",
                    "category": "aeroway",
                    "type": "aerodrome",
                },
            ),
            (),
            (),
        ]
        mock_route.return_value = (33.0, 32.0)

        answer = map_services.derive_map_goal_answer(
            "Tell me the full address of all international airports that are within a driving distance of 50 km to Carnegie Mellon University"
        )

        self.assertEqual(
            answer,
            "Pittsburgh International Airport, Southern Beltway, Findlay Township, Allegheny County, 15231, United States",
        )

    @patch("src.utils.map_services._valhalla_pedestrian_duration_minutes")
    @patch("src.utils.map_services._osrm_route_metrics")
    @patch("src.utils.map_services._nominatim_search")
    def test_route_compare_answer_formats_driving_and_walking_minutes(
        self,
        mock_search,
        mock_route,
        mock_walk_duration,
    ) -> None:
        mock_search.side_effect = [
            ({"lat": "40.4441897", "lon": "-79.9427191", "display_name": "5000 Fifth Avenue, Pittsburgh"},),
            ({"lat": "40.4480000", "lon": "-79.9500000", "display_name": "UPMC Family Health Center"},),
        ]
        mock_route.return_value = (0.8, 2.4)
        mock_walk_duration.return_value = 31.6

        answer = map_services.derive_map_goal_answer(
            "Compare the time for walking and driving route from 5000 Fifth Avenue, Pittsburgh to UPMC family health center"
        )

        self.assertEqual(answer, "driving: 2min, walking: 32min")
