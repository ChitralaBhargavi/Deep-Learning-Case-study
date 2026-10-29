"""Inventory a local DDR archive extraction without modifying dataset files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".jp2",
    ".pbm",
    ".pgm",
    ".png",
    ".pnm",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}
ANNOTATION_EXTENSIONS = {".csv", ".json", ".txt", ".xml"}
SPLIT_NAMES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}
ANNOTATION_MARKERS = {
    "annotation",
    "annotations",
    "groundtruth",
    "gt",
    "label",
    "labels",
    "mask",
    "masks",
    "seg",
    "segmentation",
}
CLASS_ALIASES = {
    "microaneurysms": ("microaneurysm", "microaneurysms", "ma"),
    "haemorrhages": ("haemorrhage", "haemorrhages", "hemorrhage", "hemorrhages", "he"),
    "hard_exudates": ("hardexudate", "hardexudates", "exudate", "exudates", "ex"),
    "soft_exudates": ("softexudate", "softexudates", "se"),
    "cotton_wool_spots": ("cottonwool", "cottonwoolspot", "cottonwoolspots", "cws"),
}
CLASS_TOKEN_TO_NAME = {
    token: name for name, aliases in CLASS_ALIASES.items() for token in aliases
}
MAX_TEXT_BYTES = 10 * 1024 * 1024
MAX_LICENSE_BYTES = 1024 * 1024


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _split_for(path: Path) -> str | None:
    found = {SPLIT_NAMES[part.lower()] for part in path.parts if part.lower() in SPLIT_NAMES}
    if len(found) == 1:
        return next(iter(found))
    return "ambiguous" if found else None


def _class_evidence(path: Path) -> list[dict[str, str]]:
    tokens = set(_tokens(str(path.with_suffix(""))))
    evidence = []
    for token, class_name in CLASS_TOKEN_TO_NAME.items():
        if token in tokens:
            evidence.append(
                {
                    "candidate_class": class_name,
                    "evidence": f"explicit filename/path token '{token}'",
                    "status": "candidate_only_not_semantically_confirmed",
                }
            )
    return evidence


def _annotation_stem(path: Path) -> str:
    parts = _tokens(path.stem)
    removable = ANNOTATION_MARKERS | set(CLASS_TOKEN_TO_NAME)
    while parts and parts[-1] in removable:
        parts.pop()
    return "_".join(parts)


def _candidate_mask_name(path: Path) -> bool:
    filename_tokens = set(_tokens(path.stem))
    annotation_directories = {
        "annotation",
        "annotations",
        "groundtruth",
        "gt",
        "label",
        "labels",
        "mask",
        "masks",
    }
    directory_tokens = {
        token for part in path.parent.parts for token in _tokens(part)
    }
    return bool(
        filename_tokens & (ANNOTATION_MARKERS | set(CLASS_TOKEN_TO_NAME))
        or directory_tokens & annotation_directories
    )


def _yolo_class_names(path: Path, root: Path) -> list[str] | None:
    current = path.parent
    while current == root or root in current.parents:
        for filename in ("classes.txt", "obj.names"):
            class_file = current / filename
            if not class_file.is_file():
                continue
            try:
                names = [
                    line.strip()
                    for line in class_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, UnicodeError):
                continue
            if names:
                return names
        if current == root:
            break
        current = current.parent
    return None


def _inspect_image(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    record: dict[str, Any] = {
        "path": relative,
        "extension": path.suffix.lower(),
        "split": _split_for(path.relative_to(root)),
        "readable": False,
    }
    try:
        record["bytes"] = path.stat().st_size
        with Image.open(path) as image:
            image.load()
            record.update(
                {
                    "readable": True,
                    "format": image.format,
                    "width": image.width,
                    "height": image.height,
                    "dimensions": [image.width, image.height],
                    "mode": image.mode,
                    "channels": len(image.getbands()),
                }
            )
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _mask_pixel_evidence(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        values = image.getcolors(maxcolors=65536)
        unique_count = len(values) if values is not None else None
        nonempty = image.convert("RGB").getbbox() is not None
        colors = []
        if values is not None:
            colors = [list(color) if isinstance(color, tuple) else color for _, color in values[:12]]
        channels = len(image.getbands())
        plausible = nonempty and (
            channels == 1 and unique_count is not None and unique_count <= 256
            or channels > 1 and unique_count is not None and unique_count <= 32
        )
        return {
            "mode": image.mode,
            "channels": channels,
            "unique_color_count": unique_count,
            "unique_values_or_colors_sample": colors,
            "non_empty_pixels": nonempty,
            "mask_like_pixel_data": plausible,
        }


def _image_indexes(
    image_records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_relative: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in image_records:
        if not record["readable"] or record.get("candidate_mask_name"):
            continue
        path = Path(record["path"])
        by_stem[path.stem.lower()].append(record)
        by_relative[record["path"].lower()].append(record)
    return by_stem, by_relative


def _match_image(
    filename: str,
    by_stem: dict[str, list[dict[str, Any]]],
    by_relative: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    normalized = filename.replace("\\", "/").lower()
    if normalized in by_relative:
        matches = by_relative[normalized]
    else:
        stem = Path(normalized).stem
        matches = by_stem.get(stem, [])
        if not matches:
            key = _annotation_stem(Path(stem))
            matches = [
                record
                for source_stem, records in by_stem.items()
                if _annotation_stem(Path(source_stem)) == key and key
            ]
    if len(matches) == 1:
        return matches[0], "unique filename/stem match"
    return None, "missing image match" if not matches else "ambiguous image match"


def _box_record(
    path: Path,
    root: Path,
    image: dict[str, Any] | None,
    status: str,
    boxes: list[dict[str, Any]],
    classes: list[str | None],
    evidence: str,
) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "format": "bounding_boxes",
        "status": status,
        "evidence": evidence,
        "matched_image": image["path"] if image else None,
        "matched_image_dimensions": image.get("dimensions") if image else None,
        "boxes": boxes,
        "candidate_classes": sorted({value for value in classes if value}),
        "class_ids_or_names_unmapped": sorted({str(value) for value in classes if value is None}),
    }


def _in_bounds(box: tuple[float, float, float, float], image: dict[str, Any]) -> bool:
    x1, y1, x2, y2 = box
    width, height = image["width"], image["height"]
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def _parse_voc(
    path: Path,
    root: Path,
    by_stem: dict[str, list[dict[str, Any]]],
    by_relative: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    try:
        xml_root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    filename = xml_root.findtext("filename")
    if not filename:
        return None
    image, match_reason = _match_image(filename, by_stem, by_relative)
    objects = xml_root.findall("object")
    if not objects:
        return None
    boxes, classes = [], []
    valid = image is not None
    for item in objects:
        name = item.findtext("name")
        node = item.find("bndbox")
        try:
            coords = tuple(float(node.findtext(key, "")) for key in ("xmin", "ymin", "xmax", "ymax"))
        except (AttributeError, ValueError):
            valid = False
            continue
        if image is None or not _in_bounds(coords, image):
            valid = False
        boxes.append({"xyxy": list(coords), "class_name": name})
        classes.append(name)
    status = "verified_pair" if valid and boxes else match_reason if image is None else "invalid_box_coordinates"
    return _box_record(path, root, image, status, boxes, classes, "XML contains object/name/bndbox fields")


def _parse_coco(
    path: Path,
    root: Path,
    data: Any,
    by_stem: dict[str, list[dict[str, Any]]],
    by_relative: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or not all(isinstance(data.get(key), list) for key in ("images", "annotations")):
        return []
    categories = {
        item.get("id"): item.get("name")
        for item in data.get("categories", [])
        if isinstance(item, dict) and "id" in item
    }
    images = {item.get("id"): item for item in data["images"] if isinstance(item, dict) and "id" in item}
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        if isinstance(annotation, dict):
            grouped[annotation.get("image_id")].append(annotation)
    output = []
    for image_id, annotations in grouped.items():
        image_info = images.get(image_id)
        if not image_info or not image_info.get("file_name"):
            continue
        image, match_reason = _match_image(image_info["file_name"], by_stem, by_relative)
        boxes, classes = [], []
        valid = image is not None
        for item in annotations:
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                valid = False
                continue
            try:
                x, y, width, height = (float(value) for value in bbox)
            except (TypeError, ValueError):
                valid = False
                continue
            coords = (x, y, x + width, y + height)
            if image is None or width <= 0 or height <= 0 or not _in_bounds(coords, image):
                valid = False
            class_name = categories.get(item.get("category_id"))
            boxes.append({"xywh": [x, y, width, height], "class_name": class_name})
            classes.append(class_name)
        status = "verified_pair" if valid and boxes else match_reason if image is None else "invalid_box_coordinates"
        output.append(
            _box_record(
                path,
                root,
                image,
                status,
                boxes,
                classes,
                "JSON contains COCO-style images, annotations, bbox, and category metadata",
            )
        )
    return output


def _parse_yolo(
    path: Path,
    root: Path,
    text: str,
    by_stem: dict[str, list[dict[str, Any]]],
    class_names: list[str] | None,
) -> dict[str, Any] | None:
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if not lines or any(len(row) != 5 for row in lines):
        return None
    boxes, classes, valid = [], [], True
    for row in lines:
        try:
            class_id = int(row[0])
            cx, cy, width, height = (float(value) for value in row[1:])
        except ValueError:
            return None
        if class_id < 0 or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            valid = False
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else None
        classes.append(class_name)
        boxes.append({"class_id": class_id, "xywh_normalized": [cx, cy, width, height]})
    image, match_reason = _match_image(path.stem, by_stem, {})
    status = "verified_pair" if valid and image else match_reason if image is None else "invalid_normalized_box"
    return _box_record(
        path,
        root,
        image,
        status,
        boxes,
        classes,
        "text rows contain YOLO class_id, normalized center_x/center_y/width/height",
    )


def _parse_csv_boxes(
    path: Path,
    root: Path,
    text: str,
    by_stem: dict[str, list[dict[str, Any]]],
    by_relative: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    try:
        rows = list(csv.DictReader(text.splitlines()))
    except csv.Error:
        return []
    if not rows or not rows[0]:
        return []
    fields = {key.strip().lower(): key for key in rows[0] if key}
    filename_key = next((fields[key] for key in ("filename", "file_name", "image", "image_id", "image_name") if key in fields), None)
    class_key = next((fields[key] for key in ("class", "class_name", "category", "label", "lesion") if key in fields), None)
    coordinate_sets = (
        ("xmin", "ymin", "xmax", "ymax"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("x1", "y1", "x2", "y2"),
    )
    coordinate_keys = next((tuple(fields[key] for key in names) for names in coordinate_sets if all(key in fields for key in names)), None)
    if not filename_key or not coordinate_keys:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(filename_key, "").strip()].append(row)
    results = []
    for filename, items in grouped.items():
        image, match_reason = _match_image(filename, by_stem, by_relative)
        boxes, classes, valid = [], [], image is not None
        for item in items:
            try:
                coords = tuple(float(item[key]) for key in coordinate_keys)
            except (TypeError, ValueError):
                valid = False
                continue
            if image is None or not _in_bounds(coords, image):
                valid = False
            boxes.append({"xyxy": list(coords), "class_name": item.get(class_key) if class_key else None})
            classes.append(item.get(class_key) if class_key else None)
        status = "verified_pair" if valid and boxes else match_reason if image is None else "invalid_box_coordinates"
        results.append(
            _box_record(path, root, image, status, boxes, classes, "CSV has image reference and explicit box-coordinate columns")
        )
    return results


def _parse_csv_image_labels(
    path: Path,
    root: Path,
    text: str,
    by_stem: dict[str, list[dict[str, Any]]],
    by_relative: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    try:
        rows = list(csv.DictReader(text.splitlines()))
    except csv.Error:
        return None
    if not rows or not rows[0]:
        return None
    fields = {key.strip().lower(): key for key in rows[0] if key}
    filename_key = next(
        (fields[key] for key in ("filename", "file_name", "image", "image_id", "image_name") if key in fields),
        None,
    )
    label_key = next(
        (
            fields[key]
            for key in ("grade", "dr_grade", "severity", "diagnosis", "disease_grade", "label", "class")
            if key in fields
        ),
        None,
    )
    if not filename_key or not label_key:
        return None
    linked_images = []
    missing_references = []
    label_values = set()
    for row in rows:
        reference = row.get(filename_key, "").strip()
        image, match_reason = _match_image(reference, by_stem, by_relative)
        if image:
            linked_images.append(image["path"])
        else:
            missing_references.append({"reference": reference, "reason": match_reason})
        if row.get(label_key) is not None:
            label_values.add(row[label_key].strip())
    return {
        "path": path.relative_to(root).as_posix(),
        "format": "image_level_labels_not_lesion_annotations",
        "status": "parsed_image_labels_not_lesion_ground_truth",
        "evidence": f"CSV has image-reference field '{filename_key}' and label field '{label_key}'",
        "linked_images": sorted(set(linked_images)),
        "missing_references": missing_references,
        "label_values_observed": sorted(label_values),
    }


def _license_notes(root: Path) -> list[dict[str, str]]:
    notes = []
    name_pattern = re.compile(r"^(license|licence|copying|notice|readme|citation)(.*)$", re.IGNORECASE)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not name_pattern.match(path.name):
            continue
        try:
            raw = path.read_bytes()[:MAX_LICENSE_BYTES]
            text = raw.decode("utf-8", errors="replace")
        except OSError as exc:
            notes.append({"path": path.relative_to(root).as_posix(), "status": f"unreadable: {exc}"})
            continue
        matches = sorted(set(re.findall(r"CC\s*BY(?:-[A-Z]+){1,3}\s*4\.0|MIT|Apache[- ]2\.0|GPL[- ]\w+", text, re.I)))
        evidence = [
            line.strip()[:300]
            for line in text.splitlines()
            if re.search(r"license|licence|attribution|noncommercial|non-commercial|sharealike|cite|citation", line, re.I)
        ][:8]
        notes.append(
            {
                "path": path.relative_to(root).as_posix(),
                "status": "read",
                "license_identifiers_found": ", ".join(matches) if matches else "no recognized license identifier found",
                "evidence_lines": " | ".join(evidence),
            }
        )
    return notes


def inventory_dataset(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")

    image_paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    image_records = [_inspect_image(path, root) for path in image_paths]
    for record in image_records:
        record["candidate_mask_name"] = _candidate_mask_name(Path(record["path"]))

    by_stem, by_relative = _image_indexes(image_records)
    annotation_records: list[dict[str, Any]] = []
    missing_files: list[dict[str, str]] = []
    mismatched_files: list[dict[str, str]] = []

    for record, path in zip(image_records, image_paths):
        if not record["candidate_mask_name"]:
            continue
        item = {
            "path": record["path"],
            "format": "candidate_mask_image",
            "status": "unreadable_image" if not record["readable"] else "unverified_candidate",
            "evidence": "annotation/class-like filename or directory token; filename alone is not verification",
            "matched_image": None,
            "matched_image_dimensions": None,
            "mask_dimensions": record.get("dimensions"),
            "pixel_evidence": None,
            "candidate_classes": _class_evidence(Path(record["path"])),
        }
        if record["readable"]:
            try:
                pixel_evidence = _mask_pixel_evidence(path)
                item["pixel_evidence"] = pixel_evidence
                key = _annotation_stem(path)
                matches = by_stem.get(key.lower(), [])
                if not matches and key:
                    matches = [
                        source for stem, sources in by_stem.items()
                        if _annotation_stem(Path(stem)) == key
                        for source in sources
                    ]
                if len(matches) != 1:
                    item["status"] = "missing_image_match" if not matches else "ambiguous_image_match"
                    missing_files.append({"path": record["path"], "reason": item["status"]})
                elif matches[0]["dimensions"] != record.get("dimensions"):
                    item["status"] = "dimension_mismatch"
                    item["matched_image"] = matches[0]["path"]
                    item["matched_image_dimensions"] = matches[0]["dimensions"]
                    mismatched_files.append(
                        {
                            "path": record["path"],
                            "image": matches[0]["path"],
                            "mask_dimensions": str(record.get("dimensions")),
                            "image_dimensions": str(matches[0]["dimensions"]),
                        }
                    )
                elif not pixel_evidence["non_empty_pixels"]:
                    item["status"] = "empty_mask_unverified"
                    item["matched_image"] = matches[0]["path"]
                    item["matched_image_dimensions"] = matches[0]["dimensions"]
                elif not pixel_evidence["mask_like_pixel_data"]:
                    item["status"] = "unverified_not_mask_like"
                    item["matched_image"] = matches[0]["path"]
                    item["matched_image_dimensions"] = matches[0]["dimensions"]
                else:
                    item["status"] = "verified_pair"
                    item["matched_image"] = matches[0]["path"]
                    item["matched_image_dimensions"] = matches[0]["dimensions"]
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                item["status"] = "unreadable_mask_pixels"
                item["error"] = f"{type(exc).__name__}: {exc}"
        annotation_records.append(item)

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ANNOTATION_EXTENSIONS:
            continue
        if path.name.lower() in {"classes.txt", "obj.names", "readme.txt", "license.txt"}:
            continue
        try:
            if path.stat().st_size > MAX_TEXT_BYTES:
                continue
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            if _candidate_mask_name(path):
                annotation_records.append(
                    {"path": path.relative_to(root).as_posix(), "format": "unknown", "status": "unreadable_annotation"}
                )
            continue

        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".xml":
            parsed = _parse_voc(path, root, by_stem, by_relative)
            if parsed:
                annotation_records.append(parsed)
        elif path.suffix.lower() == ".json":
            try:
                parsed_json = json.loads(text)
            except json.JSONDecodeError:
                parsed_json = None
            annotation_records.extend(_parse_coco(path, root, parsed_json, by_stem, by_relative))
        elif path.suffix.lower() == ".txt":
            parsed = _parse_yolo(path, root, text, by_stem, _yolo_class_names(path, root))
            if parsed:
                annotation_records.append(parsed)
        elif path.suffix.lower() == ".csv":
            parsed_boxes = _parse_csv_boxes(path, root, text, by_stem, by_relative)
            if parsed_boxes:
                annotation_records.extend(parsed_boxes)
            else:
                parsed_labels = _parse_csv_image_labels(path, root, text, by_stem, by_relative)
                if parsed_labels:
                    annotation_records.append(parsed_labels)

        if (
            not any(item.get("path") == relative for item in annotation_records)
            and _candidate_mask_name(path)
        ):
            annotation_records.append(
                {
                    "path": relative,
                    "format": "unclassified_annotation_candidate",
                    "status": "unverified_candidate",
                    "evidence": "annotation-like name, but file contents did not match a supported annotation structure",
                }
            )

    for item in annotation_records:
        if item.get("status") in {"missing image match", "missing_image_match"}:
            missing_files.append({"path": item["path"], "reason": item["status"]})
        elif item.get("status") in {"ambiguous image match", "ambiguous_image_match"}:
            mismatched_files.append({"path": item["path"], "reason": item["status"]})

    verified_images = sorted(
        {
            item["matched_image"]
            for item in annotation_records
            if item.get("status") == "verified_pair" and item.get("matched_image")
        }
    )
    annotation_file_paths = {item["path"] for item in annotation_records}
    mask_file_paths = {
        item["path"] for item in annotation_records if item.get("format") == "candidate_mask_image"
    }
    box_file_paths = {
        item["path"] for item in annotation_records if item.get("format") == "bounding_boxes"
    }
    class_summary: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in annotation_records:
        for candidate in item.get("candidate_classes", []):
            if isinstance(candidate, dict):
                class_summary[candidate["candidate_class"]].append(
                    {"file": item["path"], "evidence": candidate["evidence"], "status": candidate["status"]}
                )
        for class_name in item.get("candidate_classes", []):
            if isinstance(class_name, str):
                class_summary[class_name].append(
                    {"file": item["path"], "evidence": item.get("evidence", "parsed annotation class field"), "status": "name_from_annotation_content"}
                )
    splits: dict[str, dict[str, int]] = defaultdict(lambda: {"image_files": 0, "annotation_files": 0})
    for record in image_records:
        if record["split"]:
            splits[record["split"]]["image_files"] += 1
    annotation_paths_by_split: dict[str, set[str]] = defaultdict(set)
    for item in annotation_records:
        split = _split_for(Path(item["path"]))
        if split:
            annotation_paths_by_split[split].add(item["path"])
    for split, paths in annotation_paths_by_split.items():
        splits[split]["annotation_files"] = len(paths)

    unreadable = [
        {"path": record["path"], "reason": record.get("error", "unreadable image")}
        for record in image_records if not record["readable"]
    ]
    missing_files = list(
        {(item["path"], item["reason"]): item for item in missing_files}.values()
    )
    mismatched_files = list(
        {
            (item["path"], item.get("reason", item.get("image", ""))): item
            for item in mismatched_files
        }.values()
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "summary": {
            "image_file_count": len(image_records),
            "readable_image_count": sum(record["readable"] for record in image_records),
            "source_image_candidate_count": sum(record["readable"] and not record["candidate_mask_name"] for record in image_records),
            "annotation_file_count": len(annotation_file_paths),
            "annotation_record_count": len(annotation_records),
            "images_with_verified_annotations": len(verified_images),
            "verified_image_paths": verified_images,
            "mask_annotation_count": len(mask_file_paths),
            "bounding_box_annotation_file_count": len(box_file_paths),
            "bounding_box_annotation_record_count": sum(item.get("format") == "bounding_boxes" for item in annotation_records),
            "image_level_label_file_count": len(
                {
                    item["path"] for item in annotation_records
                    if item.get("format") == "image_level_labels_not_lesion_annotations"
                }
            ),
        },
        "splits": dict(sorted(splits.items())),
        "candidate_lesion_classes": dict(sorted(class_summary.items())),
        "images": image_records,
        "annotations": annotation_records,
        "missing_files": missing_files,
        "unreadable_files": unreadable,
        "mismatched_files": mismatched_files,
        "license_and_attribution_notes": _license_notes(root),
        "interpretation_limits": [
            "Filename tokens and folder names are evidence for candidate annotations, not proof of lesion semantics.",
            "A structurally verified image/mask pair confirms decoding, filename linkage, and matching dimensions only; it does not validate clinical annotation correctness.",
            "Empty masks may represent valid negative labels or incomplete data; their meaning cannot be inferred by this inventory.",
            "No masks, boxes, lesion classes, or missing labels are fabricated or inferred.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# DDR Dataset Archive Inventory",
        "",
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Generated (UTC): {report['generated_at_utc']}",
        f"- Image files: **{summary['image_file_count']}** ({report['summary']['readable_image_count']} readable)",
        f"- Source-image candidates: **{summary['source_image_candidate_count']}**",
        f"- Annotation files/candidates: **{summary['annotation_file_count']}**",
        f"- Images with structurally verified annotation pairs: **{summary['images_with_verified_annotations']}**",
        f"- Candidate mask files: **{summary['mask_annotation_count']}**",
        f"- Bounding-box annotation records: **{summary['bounding_box_annotation_record_count']}**",
        f"- Image-level label files (not lesion annotations): **{summary['image_level_label_file_count']}**",
        "",
        "Structural verification means the annotation could be decoded or parsed, linked to one image, and passed applicable dimension/coordinate checks. It does not verify lesion semantics or clinical correctness.",
        "",
        "## Image dimensions and formats",
        "",
        "| Dimensions | Format / mode | Files |",
        "|---|---|---:|",
    ]
    dimensions: dict[tuple[str, str], int] = defaultdict(int)
    for item in report["images"]:
        key = (
            f"{item.get('width')}×{item.get('height')}" if item["readable"] else "unreadable",
            f"{item.get('format', 'unknown')} / {item.get('mode', 'unknown')}",
        )
        dimensions[key] += 1
    lines.extend(f"| {dims} | {fmt} | {count} |" for (dims, fmt), count in sorted(dimensions.items()))
    if not dimensions:
        lines.append("| — | — | 0 |")

    lines += ["", "## Existing split directories", ""]
    if report["splits"]:
        lines += ["| Split | Image files | Annotation files/candidates |", "|---|---:|---:|"]
        lines.extend(
            f"| {name} | {counts['image_files']} | {counts['annotation_files']} |"
            for name, counts in report["splits"].items()
        )
    else:
        lines.append("No recognized train/validation/test directory names were found.")

    lines += ["", "## Candidate lesion classes and evidence", ""]
    if report["candidate_lesion_classes"]:
        lines += ["| Candidate class | Evidence | Status |", "|---|---|---|"]
        for class_name, entries in report["candidate_lesion_classes"].items():
            for entry in entries:
                lines.append(f"| {class_name} | `{entry['file']}` — {entry['evidence']} | {entry['status']} |")
    else:
        lines.append("No class mapping candidates found in parsed annotations or explicit filename/path tokens.")
    lines += [
        "",
        "Class names inferred from filenames or parsed label strings remain candidates; no uncertain aliases are silently mapped to clinical lesion classes.",
        "",
        "## Annotation inventory",
        "",
        "| File | Format | Status | Linked image / evidence |",
        "|---|---|---|---|",
    ]
    if report["annotations"]:
        for item in report["annotations"]:
            detail = item.get("matched_image") or item.get("evidence") or ""
            if item.get("mask_dimensions") is not None:
                detail = (
                    f"mask dimensions {item['mask_dimensions']}; paired image dimensions "
                    f"{item.get('matched_image_dimensions')}; {detail}"
                )
            elif item.get("matched_image_dimensions") is not None:
                detail = f"image dimensions {item['matched_image_dimensions']}; {detail}"
            if item.get("pixel_evidence"):
                pixels = item["pixel_evidence"]
                detail = (
                    f"{detail}; {pixels['mode']}, {pixels['channels']} channel(s), "
                    f"{pixels['unique_color_count']} unique colors, non-empty={pixels['non_empty_pixels']}"
                )
            lines.append(f"| `{item['path']}` | {item['format']} | {item['status']} | {detail} |")
    else:
        lines.append("| — | — | No annotation candidates found | — |")

    lines += ["", "## Missing, unreadable, and mismatched files", ""]
    for heading, key in (("Missing", "missing_files"), ("Unreadable", "unreadable_files"), ("Mismatched or ambiguous", "mismatched_files")):
        entries = report[key]
        lines.append(f"### {heading} ({len(entries)})")
        lines.append("")
        lines.extend(f"- `{entry['path']}` — {entry.get('reason', entry.get('error', ''))}" for entry in entries)
        if not entries:
            lines.append("- None detected.")
        lines.append("")

    lines += ["## License and attribution evidence", ""]
    if report["license_and_attribution_notes"]:
        for item in report["license_and_attribution_notes"]:
            lines.append(
                f"- `{item['path']}` — {item.get('license_identifiers_found', item.get('status'))}; "
                f"{item.get('evidence_lines', '')}"
            )
    else:
        lines.append("No license, notice, README, or citation files were found. Reuse terms and attribution obligations remain unverified; consult the dataset provider before use.")
    lines += [
        "",
        "## Limits and next gate",
        "",
        "- This inventory checks local file structure, not the truth or quality of lesion annotations.",
        "- Review the source license and attribution terms, annotation formats, class mapping, split integrity, and all mismatches before training.",
        "- Do not start segmentation training until the inventory is reviewed and the annotation format and reuse rights are confirmed.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Path to the extracted DDR dataset root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for ddr_inventory.json and ddr_inventory.md (must be outside dataset root)",
    )
    args = parser.parse_args(argv)
    try:
        root = args.root.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir == root or root in output_dir.parents:
            raise ValueError("Choose an --output-dir outside the dataset root to keep dataset files unchanged.")
        report = inventory_dataset(root)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ddr_inventory.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "ddr_inventory.md").write_text(render_markdown(report), encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"DDR inventory failed: {exc}", file=sys.stderr)
        return 2
    print(f"Inventory complete: {output_dir / 'ddr_inventory.json'}")
    print(f"Report: {output_dir / 'ddr_inventory.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
