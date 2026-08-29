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
    major_pitch: float,
    cross_pitch: float,
) -> np.ndarray:
    """用当前网格单元内的墨迹重心消除少量 OCR/网格相位偏差。"""

    height, width = mask.shape[:2]
    x1 = max(0, int(np.floor(np.min(polygon[:, 0]))))
    y1 = max(0, int(np.floor(np.min(polygon[:, 1]))))
    x2 = min(width, int(np.ceil(np.max(polygon[:, 0]))) + 1)
    y2 = min(height, int(np.ceil(np.max(polygon[:, 1]))) + 1)
    if x2 <= x1 or y2 <= y1:
        return polygon
    local = mask[y1:y2, x1:x2]
    ys, xs = np.nonzero(local)
    if xs.size < 4:
        return polygon
    ink_center = np.asarray((x1 + float(np.mean(xs)), y1 + float(np.mean(ys))))
    polygon_center = np.mean(polygon, axis=0)
    shift = ink_center - polygon_center
    # 字形本身可能偏旁不对称，只吸收小幅偏差，避免把网格跟着字形重心拉歪。
    limit = np.asarray((cross_pitch * 0.18, major_pitch * 0.18), dtype=np.float32)
    shift = np.clip(shift, -limit, limit)
    return polygon + shift.astype(np.float32)


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
    dominant_vertical = sum(item.vertical for item in line_items) >= len(line_items) / 2
    lines = [item for item in line_items if item.vertical == dominant_vertical]
    retained, cross_pitch = _regularize_text_lines(lines)
    if len(retained) < 2 or cross_pitch <= 0.0:
        return ()
    major_pitch, major_phase = _major_grid_geometry(
        mask,
        vertical=dominant_vertical,
        expected_pitch=cross_pitch,
        lines=retained,
    )

    height, width = rgb.shape[:2]
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
        half_cross = min(cross_length * 0.48, cross_pitch * 0.46)
        major_axis = 1 if vertical else 0
        axis_component = float(unit_major[major_axis])
        if abs(axis_component) < 0.5:
            continue
        axis_start = max(0.0, float(np.min(box[:, major_axis])))
        axis_end = min(
            float((height if vertical else width) - 1),
            float(np.max(box[:, major_axis])),
        )
        first_index = int(np.floor((axis_start - major_phase) / major_pitch)) - 1
        last_index = int(np.ceil((axis_end - major_phase) / major_pitch)) + 1
        for grid_index in range(first_index, last_index + 1):
            axis_center = major_phase + (grid_index + 0.5) * major_pitch
            if not axis_start - major_pitch * 0.18 <= axis_center <= axis_end + major_pitch * 0.18:
                continue
            distance = (axis_center - float(start_center[major_axis])) / axis_component
            center = start_center + unit_major * distance
            half_major = major_pitch * 0.46 / abs(axis_component)
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
                major_pitch=major_pitch,
                cross_pitch=cross_pitch,
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
                return tuple(output)
    return tuple(output)


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
    return tuple(output)


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
