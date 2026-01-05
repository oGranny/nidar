"""Lightweight coordinate persistence helpers for geotag-aware tooling."""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence, Tuple

# Store the database next to this module so it stays local to the app install.
DEFAULT_DB_PATH = Path(__file__).resolve().with_name("geotag.db")


def _ensure_schema(db_path: Path) -> None:
	"""Create the coordinates table the first time the database is used."""

	with sqlite3.connect(db_path) as conn:
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS coordinates (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				latitude REAL NOT NULL,
				longitude REAL NOT NULL,
				created_at TEXT NOT NULL
			)
			"""
		)


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	"""Return the distance in meters between two WGS84 coordinates."""

	radius = 6_371_000  # Earth radius in meters
	d_lat = math.radians(lat2 - lat1)
	d_lon = math.radians(lon2 - lon1)
	a = (
		math.sin(d_lat / 2) ** 2
		+ math.cos(math.radians(lat1))
		* math.cos(math.radians(lat2))
		* math.sin(d_lon / 2) ** 2
	)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	return radius * c


def _normalize_coordinate(coordinate: Sequence[float]) -> Tuple[float, float]:
	if len(coordinate) != 2:
		raise ValueError("Coordinate must contain exactly two values: latitude and longitude")
	lat, lon = float(coordinate[0]), float(coordinate[1])
	if not -90 <= lat <= 90 or not -180 <= lon <= 180:
		raise ValueError("Latitude must be in [-90, 90] and longitude in [-180, 180]")
	return lat, lon


def save_coordinate_with_threshold(
	coordinate: Sequence[float],
	threshold_meters: float,
	db_path: Path | None = None,
) -> dict:
	"""Persist a coordinate and optionally replace it with the mean of all points.

	The function always inserts a row into the local SQLite database. If the new
	coordinate is farther than ``threshold_meters`` from every point already in
	the table, the value written is the mean latitude/longitude of all points
	(including the new one). This dampens outliers while still tracking the
	history of writes.

	Returns a small dictionary summarizing the operation so callers can tell
	whether the mean was used.
	"""

	if threshold_meters <= 0:
		raise ValueError("threshold_meters must be a positive value")

	db_file = Path(db_path) if db_path else DEFAULT_DB_PATH
	_ensure_schema(db_file)

	lat, lon = _normalize_coordinate(coordinate)

	with sqlite3.connect(db_file) as conn:
		cursor = conn.execute("SELECT latitude, longitude FROM coordinates")
		existing_points = cursor.fetchall()

		use_mean = False
		if existing_points:
			distances = [
				_haversine_distance_m(lat, lon, row[0], row[1]) for row in existing_points
			]
			use_mean = all(distance > threshold_meters for distance in distances)

		if use_mean:
			total_lat = lat + sum(row[0] for row in existing_points)
			total_lon = lon + sum(row[1] for row in existing_points)
			count = len(existing_points) + 1
			lat = total_lat / count
			lon = total_lon / count

		stored_at = datetime.now(timezone.utc).isoformat()
		conn.execute(
			"INSERT INTO coordinates (latitude, longitude, created_at) VALUES (?, ?, ?)",
			(lat, lon, stored_at),
		)
		conn.commit()

		return {
			"latitude": lat,
			"longitude": lon,
			"used_mean": use_mean,
			"stored_at": stored_at,
			"db_path": str(db_file),
		}


def list_coordinates(db_path: Path | None = None) -> Iterable[Tuple[float, float]]:
	"""Utility helper to inspect the stored coordinates for debugging."""

	db_file = Path(db_path) if db_path else DEFAULT_DB_PATH
	if not db_file.exists():
		return []

	with sqlite3.connect(db_file) as conn:
		cursor = conn.execute("SELECT latitude, longitude FROM coordinates ORDER BY id ASC")
		return cursor.fetchall()
