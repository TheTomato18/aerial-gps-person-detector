"""
Geolocates detections from a drone video against its flight-log CSV.

For each sampled video frame, a detection model locates objects in image
space. The drone's recorded GPS position, altitude, and compass heading at
that moment are then used to project each detection's pixel offset into a
GPS coordinate. Detections that fall within a set distance of each other
across multiple frames are merged into a single marker, which is confirmed
only once enough frames agree on it. The result is a standalone interactive
HTML map (flight path plus detection markers), rendered with Folium.

The rhumb-line and great-circle distance formulas used for this projection
are implemented directly below to avoid an extra geospatial dependency.

Dependencies: pip install -r requirements.txt
(ultralytics is only required when passing --weights)
"""

import argparse
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import pandas as pd
import folium


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    # if detections are closer than this, combine them into a single marker
    "MIN_SEPARATION_OF_DETECTIONS_IN_METERS": 20,
    # wait until a detection is made on this many distinct frames before
    # treating it as a confirmed marker
    "MIN_DETECTIONS_TO_MAKE_VISIBLE": 3,
}

EARTH_RADIUS_M = 6371008.8  # mean Earth radius, in meters


# ---------------------------------------------------------------------------
# Geospatial helpers
# ---------------------------------------------------------------------------

def rhumb_destination(lon: float, lat: float, distance_m: float, bearing_deg: float) -> Tuple[float, float]:
    """Given a start point, a distance (meters), and a bearing (degrees),
    return the (lon, lat) reached by travelling along a rhumb line (a path
    of constant compass bearing)."""
    delta = distance_m / EARTH_RADIUS_M  # angular distance in radians
    lambda1 = math.radians(lon)
    phi1 = math.radians(lat)
    theta = math.radians(bearing_deg)

    delta_phi = delta * math.cos(theta)
    phi2 = phi1 + delta_phi

    delta_psi = math.log(math.tan(phi2 / 2 + math.pi / 4) / math.tan(phi1 / 2 + math.pi / 4))
    q = delta_phi / delta_psi if abs(delta_psi) > 1e-12 else math.cos(phi1)

    delta_lambda = delta * math.sin(theta) / q
    lambda2 = lambda1 + delta_lambda

    out_lat = math.degrees(phi2)
    out_lon = (math.degrees(lambda2) + 540) % 360 - 180  # normalise to [-180, 180)
    return out_lon, out_lat


def haversine_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two points, in kilometers."""
    r_km = EARTH_RADIUS_M / 1000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_km * c


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FoundPoint:
    location: Tuple[float, float]                     # running average (lon, lat)
    points: List[Tuple[float, float]] = field(default_factory=list)
    visible: bool = False


@dataclass
class BBox:
    x: float  # pixel-space center x of the detection
    y: float  # pixel-space center y of the detection
    w: float = 0.0
    h: float = 0.0


@dataclass
class Prediction:
    bbox: BBox
    score: float = 1.0
    label: str = ""


# ---------------------------------------------------------------------------
# Detection model interface - swap in whatever CV model you like as long as
# it implements detect(frame) -> List[Prediction]
# ---------------------------------------------------------------------------

class DetectionModel:
    def detect(self, frame) -> List[Prediction]:
        raise NotImplementedError


class YoloDetector(DetectionModel):
    """Thin wrapper so an Ultralytics YOLO model (e.g. weights produced by
    train_model.py) drops straight into this pipeline."""

    def __init__(self, weights_path: str, conf: float = 0.4):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, frame) -> List[Prediction]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        preds = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            preds.append(Prediction(
                bbox=BBox(x=cx, y=cy, w=x2 - x1, h=y2 - y1),
                score=float(box.conf[0]),
                label=self.model.names[int(box.cls[0])],
            ))
        return preds


# ---------------------------------------------------------------------------
# Flight log parsing
# ---------------------------------------------------------------------------

def read_flight_log(csv_path: str) -> pd.DataFrame:
    """Reads the flight-log CSV, stripping whitespace from column names."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def extract_video_segment(observations: pd.DataFrame) -> pd.DataFrame:
    """Returns the first continuous block of rows flagged `isVideo`: rows
    before recording starts are skipped, and collection stops at the first
    row after recording ends."""
    video_rows = []
    started = False
    for _, row in observations.iterrows():
        is_video = float(row["isVideo"]) != 0
        if not is_video:
            if started:
                break
            continue
        started = True
        video_rows.append(row)

    video_obs = pd.DataFrame(video_rows).reset_index(drop=True)
    if video_obs.empty:
        return video_obs
    video_obs["latitude"] = video_obs["latitude"].astype(float)
    video_obs["longitude"] = video_obs["longitude"].astype(float)
    return video_obs


# ---------------------------------------------------------------------------
# De-duplication (merges a new detection into an existing cluster if it's
# close enough, otherwise starts a new one)
# ---------------------------------------------------------------------------

def _merge_or_add_detection(found_points: List[FoundPoint], lon: float, lat: float) -> None:
    min_separation_km = CONFIG["MIN_SEPARATION_OF_DETECTIONS_IN_METERS"] / 1000

    for fp in found_points:
        loc_lon, loc_lat = fp.location
        if haversine_distance_km(lon, lat, loc_lon, loc_lat) < min_separation_km:
            fp.points.append((lon, lat))
            avg_lon = sum(pt[0] for pt in fp.points) / len(fp.points)
            avg_lat = sum(pt[1] for pt in fp.points) / len(fp.points)
            fp.location = (avg_lon, avg_lat)
            if len(fp.points) >= CONFIG["MIN_DETECTIONS_TO_MAKE_VISIBLE"]:
                fp.visible = True
            return

    found_points.append(FoundPoint(location=(lon, lat), points=[(lon, lat)]))


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------

def build_map(video_obs: pd.DataFrame, found_points: List[FoundPoint], output_path: str) -> int:
    """Renders the flight path and any confirmed detection markers to a
    standalone HTML file; returns the number of confirmed markers."""
    lats = video_obs["latitude"].tolist()
    lons = video_obs["longitude"].tolist()
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]

    fmap = folium.Map(location=center, zoom_start=17, tiles="OpenStreetMap")

    path_coords = list(zip(lats, lons))
    folium.PolyLine(path_coords, color="#061ACE", weight=3).add_to(fmap)

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    visible_count = 0
    for fp in found_points:
        if len(fp.points) >= CONFIG["MIN_DETECTIONS_TO_MAKE_VISIBLE"]:
            lon, lat = fp.location
            folium.Marker(
                location=[lat, lon],
                popup=f"{len(fp.points)} detections",
                icon=folium.Icon(color="orange", icon="info-sign"),
            ).add_to(fmap)
            visible_count += 1

    fmap.save(output_path)
    return visible_count


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video(video_path: str, flight_log_path: str, model: Optional[DetectionModel],
                   output_html: str, detection_interval_s: float = 0.2) -> List[FoundPoint]:
    """Runs the full pipeline: parse the flight log, detect objects in the
    video, geolocate and de-duplicate detections, and write the HTML map.
    If `model` is None, only the flight path is plotted."""
    observations = read_flight_log(flight_log_path)
    video_obs = extract_video_segment(observations)
    if video_obs.empty:
        raise ValueError("No rows flagged isVideo were found in the flight log.")

    found_points: List[FoundPoint] = []

    if model is not None:
        # Diagonal field of view of the camera, in degrees. 59° is a
        # measured value for the DJI Mavic Air; adjust for other cameras.
        fov_atan = math.tan(math.radians(59))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        last_detection_time = -detection_interval_s
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            current_time = frame_index / fps
            frame_index += 1

            # the flight log is sampled every 100ms (10 rows/sec); pull the
            # row that corresponds to this frame's timestamp in the video
            obs_index = int(current_time * 10) % len(video_obs)
            observation = video_obs.iloc[obs_index]

            lon, lat = float(observation["longitude"]), float(observation["latitude"])
            altitude_m = float(observation["ascent(feet)"]) * 0.3048
            diagonal_distance_m = altitude_m * fov_atan
            # Rebased so it combines directly with `angle` below: a detection
            # dead ahead in-frame yields angle == 90, so bearing + angle
            # recovers the drone's plain compass heading in that case.
            bearing = (float(observation["compass_heading(degrees)"]) - 90) % 360

            # Run detection at most once every `detection_interval_s` of footage.
            if current_time - last_detection_time < detection_interval_s:
                continue
            last_detection_time = current_time

            height, width = frame.shape[:2]
            predictions = model.detect(frame)

            for p in predictions:
                x_offset = p.bbox.x - width / 2
                y_offset = p.bbox.y - height / 2

                distance_from_center_px = math.hypot(x_offset, y_offset)
                diagonal_px = math.hypot(width, height)
                percent_of_diagonal = distance_from_center_px / diagonal_px
                distance_m = percent_of_diagonal * diagonal_distance_m

                # Angle from image-center to the detection, in the bearing
                # convention set up above (`or 0.000001` avoids division by zero).
                angle = math.degrees(math.atan(y_offset / (x_offset or 0.000001)))
                if x_offset >= 0:
                    angle += 180

                point_lon, point_lat = rhumb_destination(lon, lat, distance_m, (bearing + angle) % 360)
                _merge_or_add_detection(found_points, point_lon, point_lat)

        cap.release()
    else:
        print("No model provided - skipping detection, plotting flight path only.")

    visible = build_map(video_obs, found_points, output_html)
    print(f"Wrote {output_html} with {visible} marker(s) out of "
          f"{len(found_points)} candidate detection cluster(s).")
    return found_points


def main():
    parser = argparse.ArgumentParser(
        description="Geolocate objects detected in a drone video against its flight log.")
    parser.add_argument("video", help="Path to the drone video file")
    parser.add_argument("flight_log", help="Path to the flight-log CSV")
    parser.add_argument("-o", "--output", default="flight_map.html", help="Output HTML map path")
    parser.add_argument("--weights", help="Path to YOLO weights (.pt). If omitted, only the flight path is plotted.")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds of footage between detection passes")
    args = parser.parse_args()

    model = YoloDetector(args.weights, conf=args.conf) if args.weights else None
    process_video(args.video, args.flight_log, model, args.output, detection_interval_s=args.interval)


if __name__ == "__main__":
    main()
