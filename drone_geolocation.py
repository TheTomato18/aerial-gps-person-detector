"""
Geolocates detections from a drone video against its flight-log CSV.

For each sampled video frame, a detection model locates objects in image
space. The drone's recorded GPS position, altitude, and compass heading at
that moment are then used to project each detection's pixel offset into a
GPS coordinate. Detections that fall within a set distance of each other
across multiple frames are merged into a single marker, which is confirmed
only once enough frames agree on it. The result is a standalone interactive
HTML map (flight path plus detection markers), rendered with Folium, plus a
saved snapshot image of each confirmed detection for visual confirmation.

The rhumb-line and great-circle distance formulas used for this projection
are implemented directly below to avoid an extra geospatial dependency.

Dependencies: pip install -r requirements.txt
(ultralytics is only required when passing --weights)
"""

import argparse
import base64
import math
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

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
    # saved snapshots are cropped to this multiple of the detection's bounding
    # box, so the surrounding terrain is visible around the subject
    "SNAPSHOT_CONTEXT_MULTIPLE": 4.0,
    # ...but never to a window smaller than this, since aerial detections are tiny
    "SNAPSHOT_MIN_SIZE_PX": 256,
    # width of the snapshot thumbnail embedded in each map popup
    "SNAPSHOT_THUMBNAIL_WIDTH_PX": 320,
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
class Snapshot:
    """A cropped, annotated image of a single detection, held in memory until
    its cluster is confirmed and the crop is written to disk."""
    image: Any            # BGR image array (an annotated crop of the frame)
    score: float
    frame_index: int
    time_s: float


@dataclass
class FoundPoint:
    location: Tuple[float, float]                     # running average (lon, lat)
    points: List[Tuple[float, float]] = field(default_factory=list)
    visible: bool = False
    # highest-confidence snapshot seen for this cluster, and where it was saved
    snapshot: Optional[Snapshot] = None
    snapshot_path: Optional[str] = None


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

def _merge_or_add_detection(found_points: List[FoundPoint], lon: float, lat: float,
                            snapshot: Optional[Snapshot] = None) -> None:
    min_separation_km = CONFIG["MIN_SEPARATION_OF_DETECTIONS_IN_METERS"] / 1000

    for fp in found_points:
        loc_lon, loc_lat = fp.location
        if haversine_distance_km(lon, lat, loc_lon, loc_lat) < min_separation_km:
            fp.points.append((lon, lat))
            avg_lon = sum(pt[0] for pt in fp.points) / len(fp.points)
            avg_lat = sum(pt[1] for pt in fp.points) / len(fp.points)
            fp.location = (avg_lon, avg_lat)
            # keep only the clearest look at this cluster
            if snapshot is not None and (fp.snapshot is None or snapshot.score > fp.snapshot.score):
                fp.snapshot = snapshot
            if len(fp.points) >= CONFIG["MIN_DETECTIONS_TO_MAKE_VISIBLE"]:
                fp.visible = True
            return

    found_points.append(FoundPoint(location=(lon, lat), points=[(lon, lat)], snapshot=snapshot))


def _is_confirmed(fp: FoundPoint) -> bool:
    return len(fp.points) >= CONFIG["MIN_DETECTIONS_TO_MAKE_VISIBLE"]


# ---------------------------------------------------------------------------
# Detection snapshots
# ---------------------------------------------------------------------------

def crop_detection(frame, bbox: BBox):
    """Cuts a context window around `bbox` out of `frame` and draws the
    detection box onto the copy, so a reviewer can see both the subject and
    the terrain around it."""
    height, width = frame.shape[:2]
    box_w, box_h = max(bbox.w, 1.0), max(bbox.h, 1.0)

    # window is a multiple of the box, floored at a minimum size and capped
    # at the frame itself
    span = max(box_w, box_h) * CONFIG["SNAPSHOT_CONTEXT_MULTIPLE"]
    span = max(span, float(CONFIG["SNAPSHOT_MIN_SIZE_PX"]))
    span = min(span, float(min(width, height)))
    half = span / 2

    # shift the window back inside the frame rather than letting it clip, so
    # detections near an edge still get a full-size snapshot
    left = int(round(min(max(bbox.x - half, 0.0), width - span)))
    top = int(round(min(max(bbox.y - half, 0.0), height - span)))
    right = min(left + int(round(span)), width)
    bottom = min(top + int(round(span)), height)

    crop = frame[top:bottom, left:right].copy()
    if crop.size == 0:
        return None

    x1 = max(int(round(bbox.x - box_w / 2)) - left, 0)
    y1 = max(int(round(bbox.y - box_h / 2)) - top, 0)
    x2 = min(int(round(bbox.x + box_w / 2)) - left, crop.shape[1] - 1)
    y2 = min(int(round(bbox.y + box_h / 2)) - top, crop.shape[0] - 1)
    cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 140, 255), 2)
    return crop


def _format_timestamp(time_s: float) -> str:
    return f"{int(time_s // 60):02d}:{time_s % 60:04.1f}"


def save_snapshots(found_points: List[FoundPoint], output_dir: str) -> int:
    """Writes one snapshot image per confirmed detection cluster into
    `output_dir`, recording the path on each cluster. Returns the number of
    images written."""
    confirmed = [fp for fp in found_points if _is_confirmed(fp) and fp.snapshot is not None]
    if not confirmed:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    written = 0
    for i, fp in enumerate(confirmed, start=1):
        lon, lat = fp.location
        path = os.path.join(output_dir, f"detection_{i:02d}.jpg")
        if not cv2.imwrite(path, fp.snapshot.image):
            print(f"Warning: could not write snapshot {path}")
            continue
        fp.snapshot_path = path
        written += 1
        print(f"  {path}  ({lat:.6f}, {lon:.6f})  conf {fp.snapshot.score:.2f}  "
              f"at {_format_timestamp(fp.snapshot.time_s)}")
    return written


def _thumbnail_data_uri(image) -> Optional[str]:
    """Encodes a snapshot as a base64 JPEG data URI so it can be embedded
    directly in the map popup, keeping the HTML standalone."""
    width = CONFIG["SNAPSHOT_THUMBNAIL_WIDTH_PX"]
    if image.shape[1] > width:
        scale = width / image.shape[1]
        image = cv2.resize(image, (width, max(int(image.shape[0] * scale), 1)),
                           interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------

def _marker_popup_html(fp: FoundPoint) -> str:
    """Popup for one confirmed detection: its snapshot (if one was captured)
    above the coordinates and supporting evidence."""
    lon, lat = fp.location

    image_tag = ""
    if fp.snapshot is not None:
        data_uri = _thumbnail_data_uri(fp.snapshot.image)
        if data_uri:
            image_tag = f'<img src="{data_uri}" style="width:100%;border-radius:4px;">'

    lines = [f"<b>{len(fp.points)} detections</b>", f"{lat:.6f}, {lon:.6f}"]
    if fp.snapshot is not None:
        lines.append(f"best frame {fp.snapshot.frame_index} at "
                     f"{_format_timestamp(fp.snapshot.time_s)} "
                     f"(conf {fp.snapshot.score:.2f})")
    if fp.snapshot_path:
        lines.append(f"<code>{os.path.basename(fp.snapshot_path)}</code>")

    return (f'<div style="font-family:sans-serif;font-size:12px;width:320px;">'
            f'{image_tag}<div style="margin-top:6px;">{"<br>".join(lines)}</div></div>')


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
        if _is_confirmed(fp):
            lon, lat = fp.location
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(_marker_popup_html(fp), max_width=400),
                icon=folium.Icon(color="orange", icon="info-sign"),
            ).add_to(fmap)
            visible_count += 1

    fmap.save(output_path)
    return visible_count


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video(video_path: str, flight_log_path: str, model: Optional[DetectionModel],
                   output_html: str, detection_interval_s: float = 0.2,
                   snapshot_dir: Optional[str] = "detections") -> List[FoundPoint]:
    """Runs the full pipeline: parse the flight log, detect objects in the
    video, geolocate and de-duplicate detections, save a snapshot image of
    each confirmed detection, and write the HTML map. If `model` is None,
    only the flight path is plotted; if `snapshot_dir` is None, no images
    are saved (they are still embedded in the map popups)."""
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

                crop = crop_detection(frame, p.bbox)
                snapshot = None if crop is None else Snapshot(
                    image=crop,
                    score=p.score,
                    frame_index=frame_index - 1,  # frame_index was already advanced
                    time_s=current_time,
                )
                _merge_or_add_detection(found_points, point_lon, point_lat, snapshot)

        cap.release()
    else:
        print("No model provided - skipping detection, plotting flight path only.")

    if snapshot_dir:
        saved = save_snapshots(found_points, snapshot_dir)
        print(f"Saved {saved} detection snapshot(s) to {snapshot_dir}/")

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
    parser.add_argument("--snapshots", default="detections",
                        help="Directory to save an image of each confirmed detection into")
    parser.add_argument("--no-snapshots", action="store_true",
                        help="Skip writing snapshot images (they are still embedded in the map)")
    args = parser.parse_args()

    model = YoloDetector(args.weights, conf=args.conf) if args.weights else None
    process_video(args.video, args.flight_log, model, args.output,
                  detection_interval_s=args.interval,
                  snapshot_dir=None if args.no_snapshots else args.snapshots)


if __name__ == "__main__":
    main()
