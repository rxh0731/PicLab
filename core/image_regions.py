"""图片实验室文字区域候选检测与区域掩码工具。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class TextRegionCandidate:
    """一次检测得到的文字区域，坐标相对于输入图像归一化。"""

    polygon: tuple[tuple[float, float], ...]
    confidence: float
    color: str


@dataclass(frozen=True, slots=True)
class _DetectedTextLine:
    """OCR 返回的一条文字行及其稳定几何量。"""

    box: np.ndarray
    score: float
    vertical: bool
    center: float
    major_length: float
    minor_length: float


def _resolve_polygon_overlaps(
    candidates: list[TextRegionCandidate],
    width: int,
    height: int,
    foreground_mask: np.ndarray | None = None,
) -> tuple[TextRegionCandidate, ...]:
    """消除候选区域交叠，使一个像素只归属于一个文字区域。"""

    if len(candidates) < 2:
        return tuple(candidates)
    masks: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    overlap_found = False
    for candidate in candidates:
        points = np.rint(np.asarray(candidate.polygon, dtype=np.float32) * (width, height)).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
        masks.append(mask)
        centers.append(np.mean(points.astype(np.float32), axis=0) if len(points) else np.zeros(2, np.float32))
    occupancy = np.sum(np.stack(masks, axis=0), axis=0)
    overlap_found = bool(np.any(occupancy > 1))
    if not overlap_found:
        return tuple(candidates)
    owner = np.full((height, width), -1, dtype=np.int16)
    overlap_pixels = occupancy > 1
    for index, mask in enumerate(masks):
        only = (mask > 0) & ~overlap_pixels
        owner[only] = index
    ys, xs = np.nonzero(overlap_pixels)
    if len(xs):
        pixel_points = np.column_stack((xs, ys)).astype(np.float32)
        for start in range(0, len(pixel_points), 20000):
            batch = pixel_points[start : start + 20000]
            distances = np.stack(
                [np.sum((batch - center[None, :]) ** 2, axis=1) for center in centers],
                axis=1,
            )
            active = np.stack([mask[ys[start : start + len(batch)], xs[start : start + len(batch)]] > 0 for mask in masks], axis=1)
            distances[~active] = np.inf
            owner[ys[start : start + len(batch)], xs[start : start + len(batch)]] = np.argmin(distances, axis=1)
    resolved: list[TextRegionCandidate] = []
    for index, candidate in enumerate(candidates):
        assigned = (owner == index).astype(np.uint8)
        original = masks[index]
        if foreground_mask is not None:
            original_ink = (original > 0) & (foreground_mask > 0)
            assigned_ink = (assigned > 0) & (foreground_mask > 0)
            # 不能为了消歧而裁掉该区域自己的笔画；这种情况下保留原边界，
            # 后续人工复核仍可继续调整。
            if np.count_nonzero(original_ink) and np.count_nonzero(assigned_ink) < np.count_nonzero(original_ink):
                resolved.append(candidate)
                continue
        contours, _ = cv2.findContours(assigned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        if contour is None or cv2.contourArea(contour) < 4.0:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.02), True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        normalized = tuple(
            (
                float(np.clip(point[0] / max(1, width - 1), 0.0, 1.0)),
                float(np.clip(point[1] / max(1, height - 1), 0.0, 1.0)),
            )
            for point in polygon
        )
        resolved.append(TextRegionCandidate(normalized, candidate.confidence, candidate.color))
    return _separate_polygon_bounds(resolved, width, height)


def _separate_polygon_bounds(
    candidates: list[TextRegionCandidate] | tuple[TextRegionCandidate, ...],
    width: int,
    height: int,
) -> tuple[TextRegionCandidate, ...]:
    """按相邻区域中心切开仍相交的安全边距，避免框线互相穿过。"""

    if len(candidates) < 2:
        return tuple(candidates)
    polygons = [np.asarray(item.polygon, dtype=np.float32) * (width, height) for item in candidates]
    for index in range(len(polygons)):
        for other_index in range(index + 1, len(polygons)):
            first, second = polygons[index], polygons[other_index]
            first_left, first_top = np.min(first, axis=0)
            first_right, first_bottom = np.max(first, axis=0)
            second_left, second_top = np.min(second, axis=0)
            second_right, second_bottom = np.max(second, axis=0)
            overlap_x = min(first_right, second_right) - max(first_left, second_left)
            overlap_y = min(first_bottom, second_bottom) - max(first_top, second_top)
            if overlap_x <= 0.0 or overlap_y <= 0.0:
                continue
            first_center = np.mean(first, axis=0)
            second_center = np.mean(second, axis=0)
            if overlap_x <= overlap_y:
                separator = float((first_center[0] + second_center[0]) * 0.5)
                if first_center[0] <= second_center[0]:
                    first[:, 0] = np.minimum(first[:, 0], separator + 0.49)
                    second[:, 0] = np.maximum(second[:, 0], separator - 0.51)
                else:
                    first[:, 0] = np.maximum(first[:, 0], separator - 0.51)
                    second[:, 0] = np.minimum(second[:, 0], separator + 0.49)
            else:
                separator = float((first_center[1] + second_center[1]) * 0.5)
                if first_center[1] <= second_center[1]:
                    first[:, 1] = np.minimum(first[:, 1], separator + 0.49)
                    second[:, 1] = np.maximum(second[:, 1], separator - 0.51)
                else:
                    first[:, 1] = np.maximum(first[:, 1], separator - 0.51)
                    second[:, 1] = np.minimum(second[:, 1], separator + 0.49)
    separated: list[TextRegionCandidate] = []
    for candidate, polygon in zip(candidates, polygons):
        separated.append(
            TextRegionCandidate(
                tuple(
                    (
                        float(np.clip(point[0] / max(1, width - 1), 0.0, 1.0)),
                        float(np.clip(point[1] / max(1, height - 1), 0.0, 1.0)),
                    )
                    for point in polygon
                ),
                candidate.confidence,
                candidate.color,
            )
        )
    return tuple(separated)


@lru_cache(maxsize=1)
def _rapidocr_engine() -> object:
    """延迟创建文字行检测器，避免图片实验室启动时加载模型。"""

    from rapidocr import RapidOCR

    logging.getLogger("RapidOCR").setLevel(logging.CRITICAL)
    return RapidOCR(
        params={
            "Global.log_level": "critical",
            "Global.use_cls": False,
            "Global.use_rec": False,
        }
    )


def _regularize_text_lines(
    lines: list[_DetectedTextLine],
) -> tuple[list[_DetectedTextLine], float]:
    """去除重复文字行，并把行中心吸附到统一的版面网格。"""

    if not lines:
        return [], 0.0
    typical_minor = float(np.median([line.minor_length for line in lines]))
    duplicate_distance = max(3.0, typical_minor * 0.34)
    retained: list[_DetectedTextLine] = []
    for line in sorted(lines, key=lambda item: (-item.score, item.center)):
        if any(abs(line.center - item.center) < duplicate_distance for item in retained):
            continue
        retained.append(line)
    retained.sort(key=lambda item: item.center)
    if len(retained) < 2:
        return retained, max(6.0, min(96.0, typical_minor))

    centers = np.asarray([line.center for line in retained], dtype=np.float64)
    gaps = np.diff(centers)
    plausible = gaps[
        (gaps >= max(4.0, typical_minor * 0.52))
        & (gaps <= max(8.0, typical_minor * 1.72))
    ]
    valid_gaps = gaps[gaps >= 4.0]
    if plausible.size:
        pitch = float(np.median(plausible))
    elif valid_gaps.size:
        pitch = float(np.percentile(valid_gaps, 40.0))
    else:
        pitch = typical_minor
    pitch = max(6.0, min(96.0, pitch))

    indices = np.zeros(len(centers), dtype=np.int32)
    for index in range(1, len(centers)):
        step = max(1, int(round((centers[index] - centers[index - 1]) / pitch)))
        indices[index] = indices[index - 1] + step
    if len(centers) >= 3 and int(indices[-1]) > 0:
        fitted_pitch, fitted_origin = np.polyfit(indices, centers, 1)
        if pitch * 0.78 <= fitted_pitch <= pitch * 1.22:
            pitch = float(fitted_pitch)
            origin = float(fitted_origin)
        else:
            origin = float(np.median(centers - indices * pitch))
    else:
        origin = float(np.median(centers - indices * pitch))

    by_index = {int(index): line for index, line in zip(indices, retained)}
    completed: list[_DetectedTextLine] = []
    for grid_index in range(int(indices[0]), int(indices[-1]) + 1):
        target_center = origin + grid_index * pitch
        line = by_index.get(grid_index)
        if line is None:
            nearest = min(retained, key=lambda item: abs(item.center - target_center))
            score = max(0.35, nearest.score * 0.78)
            box = np.array(nearest.box, dtype=np.float32, copy=True)
            vertical = nearest.vertical
            major_length = nearest.major_length
            minor_length = nearest.minor_length
        else:
            score = line.score
            box = np.array(line.box, dtype=np.float32, copy=True)
            vertical = line.vertical
            major_length = line.major_length
            minor_length = line.minor_length
        axis = 0 if vertical else 1
        current_center = float(np.mean(box[:, axis]))
        box[:, axis] += target_center - current_center
        completed.append(
            _DetectedTextLine(
                box=box,
                score=score,
                vertical=vertical,
                center=float(target_center),
                major_length=major_length,
                minor_length=minor_length,
            )
        )
    return completed, pitch


def _major_grid_geometry(
    mask: np.ndarray,
    *,
    vertical: bool,
    expected_pitch: float,
    lines: list[_DetectedTextLine],
) -> tuple[float, float]:
    """从整页墨迹投影估计字符方向的节距和公共边界相位。"""

    projection = np.mean(mask, axis=1 if vertical else 0).astype(np.float32)
    if projection.size < 8 or float(np.ptp(projection)) < 0.002:
        start = min(float(np.min(line.box[:, 1 if vertical else 0])) for line in lines)
        return expected_pitch, start % expected_pitch
    smoothed = cv2.GaussianBlur(
        projection.reshape(-1, 1),
        (1, 0),
        sigmaX=0.0,
        sigmaY=max(0.8, expected_pitch * 0.032),
    ).reshape(-1)
    axis = 1 if vertical else 0
    start = max(0.0, min(float(np.min(line.box[:, axis])) for line in lines))
    end = min(
        float(projection.size - 1),
        max(float(np.max(line.box[:, axis])) for line in lines),
    )
    deviation_scale = max(0.001, float(np.std(smoothed)))
    pitch_steps = max(9, int(round(expected_pitch * 1.2)))
    pitches = np.linspace(expected_pitch * 0.86, expected_pitch * 1.14, pitch_steps)
    best: tuple[float, float, float] | None = None
    sample_axis = np.arange(projection.size, dtype=np.float32)
    for pitch in pitches:
        phase_steps = max(12, int(round(pitch * 2.0)))
        for phase in np.linspace(0.0, pitch, phase_steps, endpoint=False):
            first_index = int(np.floor((start - phase) / pitch)) - 1
            last_index = int(np.ceil((end - phase) / pitch)) + 1
            boundaries = phase + np.arange(first_index, last_index + 1) * pitch
            boundaries = boundaries[(boundaries >= start) & (boundaries <= end)]
            if boundaries.size < 3:
                continue
            centers = boundaries[:-1] + pitch * 0.5
            boundary_values = []
            for offset in (-1.0, 0.0, 1.0):
                boundary_values.append(
                    np.interp(boundaries + offset, sample_axis, smoothed)
                )
            boundary_cost = float(np.mean(boundary_values))
            center_level = (
                float(np.mean(np.interp(centers, sample_axis, smoothed)))
                if centers.size
                else boundary_cost
            )
            deviation = abs(float(pitch) - expected_pitch) / expected_pitch
            cost = boundary_cost - center_level * 0.12 + deviation * deviation_scale * 0.22
            candidate = (cost, float(pitch), float(phase))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return expected_pitch, start % expected_pitch
    return best[1], best[2]


def _refine_cell_to_ink(
    polygon: np.ndarray,
    mask: np.ndarray,
    *,
    center: np.ndarray,
    unit_major: np.ndarray,
    unit_cross: np.ndarray,
    major_pitch: float,
    cross_pitch: float,
    major_scale: float,
) -> np.ndarray:
    """从网格单元内的墨迹生成带留白的多边形边界。"""

    height, width = mask.shape[:2]
    base_half_major = major_pitch * 0.54 / max(0.5, major_scale)
    base_half_cross = cross_pitch * 0.54
    envelope = np.asarray(
        [
            center - unit_major * base_half_major - unit_cross * base_half_cross,
            center - unit_major * base_half_major + unit_cross * base_half_cross,
            center + unit_major * base_half_major + unit_cross * base_half_cross,
            center + unit_major * base_half_major - unit_cross * base_half_cross,
        ],
        dtype=np.float32,
    )
    x1 = max(0, int(np.floor(np.min(envelope[:, 0]))))
    y1 = max(0, int(np.floor(np.min(envelope[:, 1]))))
    x2 = min(width, int(np.ceil(np.max(envelope[:, 0]))) + 1)
    y2 = min(height, int(np.ceil(np.max(envelope[:, 1]))) + 1)
    if x2 <= x1 or y2 <= y1:
        return polygon
    ys, xs = np.nonzero(mask[y1:y2, x1:x2])
    if xs.size < 4:
        return polygon
    points = np.column_stack((x1 + xs.astype(np.float32), y1 + ys.astype(np.float32)))
    relative = points - center.reshape(1, 2)
    major_projection = relative @ unit_major.reshape(2, 1)
    cross_projection = relative @ unit_cross.reshape(2, 1)
    major_projection = major_projection.reshape(-1)
    cross_projection = cross_projection.reshape(-1)
    inside = (
        (np.abs(major_projection) <= base_half_major)
        & (np.abs(cross_projection) <= base_half_cross)
    )
    if not np.any(inside):
        return polygon
    ink_points = points[inside]
    padding = max(2.0, min(8.0, min(major_pitch, cross_pitch) * 0.12))
    hull = cv2.convexHull(np.rint(ink_points).astype(np.int32)).reshape(-1, 2)
    if hull.shape[0] < 3:
        return polygon
    hull_center = np.mean(hull.astype(np.float32), axis=0)
    directions = hull.astype(np.float32) - hull_center
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    expanded = hull.astype(np.float32) + directions / np.maximum(lengths, 1.0) * padding
    approximation = cv2.approxPolyDP(
        expanded.reshape(-1, 1, 2),
        max(0.8, padding * 0.32),
        True,
    ).reshape(-1, 2)
    if approximation.shape[0] < 3:
        approximation = expanded
    # 保留一个网格范围内的上限，避免前景噪声把相邻字符整块连进来。
    relative = approximation - center.reshape(1, 2)
    major_projection = relative @ unit_major.reshape(2, 1)
    cross_projection = relative @ unit_cross.reshape(2, 1)
    major_projection = np.clip(
        major_projection.reshape(-1),
        -base_half_major,
        base_half_major,
    )
    cross_projection = np.clip(
        cross_projection.reshape(-1),
        -base_half_cross,
        base_half_cross,
    )
    return np.asarray(
        [
            center + unit_major * major_value + unit_cross * cross_value
            for major_value, cross_value in zip(major_projection, cross_projection)
        ],
        dtype=np.float32,
    )


def _line_guided_candidates(
    rgb: np.ndarray,
    mask: np.ndarray,
    max_regions: int,
) -> tuple[TextRegionCandidate, ...]:
    """先检测文字行，再按行距切分字符单元，适配横排和竖排文献。"""

    try:
        result = _rapidocr_engine()(rgb, use_rec=False, use_cls=False)
        boxes = np.asarray(getattr(result, "boxes", ()), dtype=np.float32)
        scores = tuple(float(value) for value in getattr(result, "scores", ()))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return ()
    if boxes.ndim != 3 or boxes.shape[1:] != (4, 2) or not scores:
        return ()
    line_items: list[_DetectedTextLine] = []
    height, width = rgb.shape[:2]
    for box, score in zip(boxes, scores):
        horizontal_length = (
            np.linalg.norm(box[1] - box[0]) + np.linalg.norm(box[2] - box[3])
        ) / 2.0
        vertical_length = (
            np.linalg.norm(box[3] - box[0]) + np.linalg.norm(box[2] - box[1])
        ) / 2.0
        vertical = vertical_length > horizontal_length
        major = max(horizontal_length, vertical_length)
        minor = min(horizontal_length, vertical_length)
        if major < 24.0 or minor < 3.0 or major / max(1.0, minor) < 2.2:
            continue
        # OCR 检测器有时会把页面边框、横线或跨栏噪声当成文字行。
        # 只有框内确实存在足够墨迹时，才允许它参与整页网格推断。
        left = max(0, int(np.floor(np.min(box[:, 0]))))
        top = max(0, int(np.floor(np.min(box[:, 1]))))
        right = min(width, int(np.ceil(np.max(box[:, 0]))) + 1)
        bottom = min(height, int(np.ceil(np.max(box[:, 1]))) + 1)
        if right <= left or bottom <= top:
            continue
        box_mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
        shifted = np.rint(box - (left, top)).astype(np.int32)
        cv2.fillPoly(box_mask, [shifted], 1)
        support = float(np.count_nonzero(mask[top:bottom, left:right] & (box_mask > 0)))
        box_area = float(np.count_nonzero(box_mask))
        if support / max(1.0, box_area) < 0.018:
            continue
        center = float(np.mean(box[:, 0] if vertical else box[:, 1]))
        line_items.append(
            _DetectedTextLine(
                box=np.array(box, dtype=np.float32, copy=True),
                score=score,
                vertical=vertical,
                center=center,
                major_length=major,
                minor_length=minor,
            )
        )
    if len(line_items) < 2:
        return ()
    # 方向按覆盖长度加权，避免少量跨栏横框压过真正的竖排文字列。
    orientation_weight = {
        value: sum(item.major_length * max(0.2, item.score) for item in line_items if item.vertical == value)
        for value in (False, True)
    }
    dominant_vertical = orientation_weight[True] > orientation_weight[False]
    dominant_weight = orientation_weight[dominant_vertical]
    total_weight = sum(orientation_weight.values())
    if total_weight <= 0.0 or dominant_weight / total_weight < 0.62:
        return ()
    lines = [item for item in line_items if item.vertical == dominant_vertical]
    if len(lines) < 2:
        return ()
    typical_minor = float(np.median([item.minor_length for item in lines]))
    typical_major = float(np.median([item.major_length for item in lines]))
    lines = [
        item
        for item in lines
        if 0.48 * typical_minor <= item.minor_length <= 2.1 * typical_minor
        and item.major_length >= max(24.0, typical_major * 0.42)
    ]
    retained, cross_pitch = _regularize_text_lines(lines)
    if len(retained) < 2 or cross_pitch <= 0.0:
        return ()
    major_pitch, major_phase = _major_grid_geometry(
        mask,
        vertical=dominant_vertical,
        expected_pitch=cross_pitch,
        lines=retained,
    )

    output: list[TextRegionCandidate] = []
    for line in retained:
        box = line.box
        line_score = line.score
        vertical = line.vertical
        if vertical:
            start_center = (box[0] + box[1]) / 2.0
            end_center = (box[3] + box[2]) / 2.0
            cross_vector = ((box[1] - box[0]) + (box[2] - box[3])) / 2.0
        else:
            start_center = (box[0] + box[3]) / 2.0
            end_center = (box[1] + box[2]) / 2.0
            cross_vector = ((box[3] - box[0]) + (box[2] - box[1])) / 2.0
        major_vector = end_center - start_center
        line_length = float(np.linalg.norm(major_vector))
        if line_length <= 1.0:
            continue
        unit_major = major_vector / line_length
        cross_length = float(np.linalg.norm(cross_vector))
        if cross_length <= 1.0:
            continue
        unit_cross = cross_vector / cross_length
        half_cross = min(cross_length * 0.52, cross_pitch * 0.52)
        major_axis = 1 if vertical else 0
        axis_component = float(unit_major[major_axis])
        if abs(axis_component) < 0.5:
            continue
        axis_start = max(0.0, float(np.min(box[:, major_axis])))
        axis_end = min(
            float((height if vertical else width) - 1),
            float(np.max(box[:, major_axis])),
        )
        # 用当前文字行的投影重新寻找网格相位。整页投影容易被标题、页边线和
        # 跨栏噪声影响，局部相位校正可避免框线在一行内逐格偏移。
        line_left = max(0, int(np.floor(np.min(box[:, 0]))))
        line_top = max(0, int(np.floor(np.min(box[:, 1]))))
        line_right = min(width, int(np.ceil(np.max(box[:, 0]))) + 1)
        line_bottom = min(height, int(np.ceil(np.max(box[:, 1]))) + 1)
        if vertical:
            line_projection = np.mean(mask[line_top:line_bottom, :], axis=1).astype(np.float32)
        else:
            line_projection = np.mean(mask[:, line_left:line_right], axis=0).astype(np.float32)
        if (not vertical) and line_projection.size >= 5 and float(np.ptp(line_projection)) > 0.01:
            smoothed = cv2.GaussianBlur(
                line_projection.reshape(-1, 1),
                (1, 0),
                sigmaX=0.0,
                sigmaY=max(0.7, major_pitch * 0.045),
            ).reshape(-1)
            sample_axis = np.arange(smoothed.size, dtype=np.float32)
            phase_candidates = major_phase + np.linspace(-0.30, 0.30, 13) * major_pitch
            best_phase = major_phase
            best_cost = float("inf")
            for phase in phase_candidates:
                first = int(np.floor((axis_start - phase) / major_pitch)) - 1
                last = int(np.ceil((axis_end - phase) / major_pitch)) + 1
                boundaries = phase + np.arange(first, last + 1) * major_pitch
                boundaries = boundaries[(boundaries >= axis_start) & (boundaries <= axis_end)]
                if boundaries.size < 2:
                    continue
                centers = boundaries[:-1] + major_pitch * 0.5
                boundary_level = float(np.mean(np.interp(boundaries, sample_axis, smoothed)))
                center_level = float(np.mean(np.interp(centers, sample_axis, smoothed)))
                cost = boundary_level - center_level * 0.26
                if cost < best_cost:
                    best_cost = cost
                    best_phase = float(phase)
            major_phase_for_line = best_phase
        else:
            major_phase_for_line = major_phase
        first_index = int(np.floor((axis_start - major_phase_for_line) / major_pitch)) - 1
        last_index = int(np.ceil((axis_end - major_phase_for_line) / major_pitch)) + 1
        for grid_index in range(first_index, last_index + 1):
            axis_center = major_phase_for_line + (grid_index + 0.5) * major_pitch
            if not axis_start - major_pitch * 0.18 <= axis_center <= axis_end + major_pitch * 0.18:
                continue
            distance = (axis_center - float(start_center[major_axis])) / axis_component
            center = start_center + unit_major * distance
            half_major = major_pitch * 0.52 / abs(axis_component)
            polygon_array = np.asarray(
                [
                    center - unit_major * half_major - unit_cross * half_cross,
                    center - unit_major * half_major + unit_cross * half_cross,
                    center + unit_major * half_major + unit_cross * half_cross,
                    center + unit_major * half_major - unit_cross * half_cross,
                ],
                dtype=np.float32,
            )
            polygon_array = _refine_cell_to_ink(
                polygon_array,
                mask,
                center=center,
                unit_major=unit_major,
                unit_cross=unit_cross,
                major_pitch=major_pitch,
                cross_pitch=cross_pitch,
                major_scale=abs(axis_component),
            )
            x1 = max(0, int(np.floor(np.min(polygon_array[:, 0]))))
            y1 = max(0, int(np.floor(np.min(polygon_array[:, 1]))))
            x2 = min(width, int(np.ceil(np.max(polygon_array[:, 0]))) + 1)
            y2 = min(height, int(np.ceil(np.max(polygon_array[:, 1]))) + 1)
            if x2 <= x1 or y2 <= y1:
                continue
            local_mask = mask[y1:y2, x1:x2]
            ink_ratio = float(np.mean(local_mask)) if local_mask.size else 0.0
            if ink_ratio < 0.012:
                continue
            local = rgb[y1:y2, x1:x2]
            mean_rgb = tuple(float(value) for value in np.mean(local.reshape(-1, 3), axis=0))
            confidence = float(
                np.clip(0.58 + line_score * 0.34 + min(0.08, ink_ratio * 0.30), 0.0, 0.98)
            )
            polygon = tuple(
                (
                    float(np.clip(point[0] / max(1, width), 0.0, 1.0)),
                    float(np.clip(point[1] / max(1, height), 0.0, 1.0)),
                )
                for point in polygon_array
            )
            output.append(TextRegionCandidate(polygon, confidence, _overlay_color(mean_rgb)))
            if len(output) >= max_regions:
                return _resolve_polygon_overlaps(output, width, height, mask)
    if not output:
        return ()
    # 网格结果必须覆盖 OCR 文字行中的主要前景；覆盖率过低通常意味着
    # OCR 行框来自页边线或跨栏误识别，此时交给连通域回退更可靠。
    line_area_mask = np.zeros((height, width), dtype=np.uint8)
    for line in retained:
        points = np.rint(line.box).astype(np.int32)
        cv2.fillPoly(line_area_mask, [points], 1)
    covered = np.zeros((height, width), dtype=np.uint8)
    for candidate in output:
        points = np.rint(np.asarray(candidate.polygon) * (width, height)).astype(np.int32)
        cv2.fillPoly(covered, [points], 1)
    relevant = mask & (line_area_mask > 0)
    relevant_count = int(np.count_nonzero(relevant))
    if relevant_count and int(np.count_nonzero(relevant & (covered > 0))) / relevant_count < 0.58:
        return ()
    return _resolve_polygon_overlaps(output, width, height, mask)


def _normalize_rgb(source: np.ndarray) -> np.ndarray:
    values = np.asarray(source)
    if values.ndim == 2:
        values = cv2.cvtColor(values.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("文字区域检测需要灰度或 RGB 图像。")
    return np.ascontiguousarray(values, dtype=np.uint8)


def _overlay_color(mean_rgb: tuple[float, float, float]) -> str:
    """从高对比度候选色中选择叠加线颜色，不修改原图颜色。"""

    candidates = (
        ("#28a6c1", (40, 166, 193)),
        ("#d95368", (217, 83, 104)),
        ("#e0a522", (224, 165, 34)),
        ("#57a773", (87, 167, 115)),
        ("#855cc7", (133, 92, 199)),
    )
    background = np.asarray(mean_rgb, dtype=np.float32)
    best_name = candidates[0][0]
    best_score = -1.0
    for name, color in candidates:
        distance = float(np.linalg.norm(background - np.asarray(color)))
        luminance_gap = abs(
            float(np.dot(background, (0.299, 0.587, 0.114)))
            - float(np.dot(color, (0.299, 0.587, 0.114)))
        )
        score = distance + luminance_gap * 0.8
        if score > best_score:
            best_name = name
            best_score = score
    return best_name


def detect_text_regions(
    source: np.ndarray,
    foreground_mask: np.ndarray | None = None,
    *,
    max_regions: int = 3000,
) -> tuple[TextRegionCandidate, ...]:
    """检测文字候选区域。

    这里不尝试识别文字内容，只生成供用户复核的几何候选区域。
    检测使用当前清理算法的前景结果，并通过轻量形态学操作合并同一字的笔画。
    """

    rgb = _normalize_rgb(source)
    height, width = rgb.shape[:2]
    if height < 8 or width < 8:
        return ()
    if foreground_mask is None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            max(15, min(101, (min(height, width) // 20) | 1)),
            8,
        ) > 0
    else:
        values = np.asarray(foreground_mask)
        if values.shape != (height, width):
            raise ValueError("文字区域检测掩码尺寸不匹配。")
        mask = values > 0
    mask_u8 = mask.astype(np.uint8) * 255
    line_candidates = _line_guided_candidates(rgb, mask, max_regions)
    if line_candidates:
        return line_candidates
    kernel_size = max(3, min(15, int(round(min(height, width) / 220.0)) | 1))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    raw_count, _raw_labels, raw_stats, _raw_centroids = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )
    image_area = float(height * width)
    minimum_area = max(20.0, image_area * 0.000012)
    spans = [
        max(
            int(raw_stats[label, cv2.CC_STAT_WIDTH]),
            int(raw_stats[label, cv2.CC_STAT_HEIGHT]),
        )
        for label in range(1, raw_count)
        if int(raw_stats[label, cv2.CC_STAT_AREA]) >= max(4, minimum_area * 0.2)
        and int(raw_stats[label, cv2.CC_STAT_AREA]) <= image_area * 0.02
    ]
    typical_span = float(np.median(spans)) if spans else float(kernel_size * 2)
    grouping_radius = max(
        kernel_size,
        min(31, int(round(typical_span * 0.46)) | 1),
    )
    grouped_mask = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (grouping_radius, grouping_radius),
        ),
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        grouped_mask,
        connectivity=8,
    )
    candidates: list[tuple[int, int, int, int, int, float]] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area or component_width < 4 or component_height < 4:
            continue
        if component_width * component_height > image_area * 0.16:
            continue
        fill_ratio = area / max(1.0, component_width * component_height)
        confidence = float(
            np.clip(
                0.45
                + min(0.35, fill_ratio * 0.45)
                + min(0.18, area / max(1.0, image_area) * 12_000.0),
                0.0,
                0.98,
            )
        )
        candidates.append((y, x, component_width, component_height, label, confidence))
    candidates.sort(key=lambda item: (item[0], item[1]))
    candidates = candidates[: max(1, int(max_regions))]

    output: list[TextRegionCandidate] = []
    for y, x, component_width, component_height, label, confidence in candidates:
        grouped_component = labels == label
        component = (grouped_component & (mask_u8 > 0)).astype(np.uint8) * 255
        contours, _hierarchy = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = max(contours, key=cv2.contourArea) if contours else None
        polygon: np.ndarray
        if contour is None or cv2.contourArea(contour) < 4.0:
            polygon = np.asarray(
                [[x, y], [x + component_width, y], [x + component_width, y + component_height], [x, y + component_height]],
                dtype=np.float32,
            )
        else:
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.025), True)
            polygon = approximation.reshape(-1, 2).astype(np.float32)
            if polygon.shape[0] < 3:
                polygon = np.asarray(
                    [[x, y], [x + component_width, y], [x + component_width, y + component_height], [x, y + component_height]],
                    dtype=np.float32,
                )
        local = rgb[y : y + component_height, x : x + component_width]
        mean_rgb = tuple(float(value) for value in np.mean(local.reshape(-1, 3), axis=0))
        normalized = tuple(
            (
                float(np.clip(point[0] / max(1, width - 1), 0.0, 1.0)),
                float(np.clip(point[1] / max(1, height - 1), 0.0, 1.0)),
            )
            for point in polygon
        )
        output.append(
            TextRegionCandidate(
                polygon=normalized,
                confidence=confidence,
                color=_overlay_color(mean_rgb),
            )
        )
    return _resolve_polygon_overlaps(output, width, height, mask)


def region_mask(
    shape: tuple[int, int],
    regions: object,
    *,
    statuses: set[str] | frozenset[str] | None = None,
    source_region: tuple[int, int, int, int] | None = None,
    source_size: tuple[int, int] | None = None,
    margin: float = 0.0,
) -> np.ndarray:
    """把项目中的归一化多边形栅格化到指定图像区域。"""

    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        return np.zeros((max(0, height), max(0, width)), dtype=bool)
    if source_region is None:
        left, top, right, bottom = 0, 0, width, height
        source_width, source_height = width, height
    else:
        if source_size is None:
            raise ValueError("区域掩码缺少原图尺寸。")
        left, top, right, bottom = (int(value) for value in source_region)
        source_width, source_height = (int(value) for value in source_size)
    result = np.zeros((height, width), dtype=np.uint8)
    selected = statuses or {"pending", "confirmed", "processed"}
    for region in regions or ():
        if str(getattr(region, "status", "pending")) not in selected:
            continue
        points = getattr(region, "polygon", ())
        if len(points) < 3:
            continue
        polygon = np.asarray(
            [
                (
                    (float(point[0]) * source_width - left)
                    * width
                    / max(1, right - left),
                    (float(point[1]) * source_height - top)
                    * height
                    / max(1, bottom - top),
                )
                for point in points
            ],
            dtype=np.float32,
        )
        if margin > 0.0:
            center = np.mean(polygon, axis=0)
            polygon = center + (polygon - center) * (1.0 + float(margin))
        cv2.fillPoly(result, [np.rint(polygon).astype(np.int32)], 255)
    return result > 0
