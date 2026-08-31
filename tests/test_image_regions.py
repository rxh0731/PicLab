"""图片实验室文字区域检测测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.image_regions import detect_text_regions, region_mask
from data.image_lab_project_store import ImageLabRegion


class ImageRegionTests(unittest.TestCase):
    def test_detect_regions_returns_normalized_candidates_and_contrast_color(self) -> None:
        source = np.full((240, 360, 3), 232, dtype=np.uint8)
        cv2.putText(source, "AB", (45, 155), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (35, 45, 55), 8, cv2.LINE_AA)
        mask = np.zeros((240, 360), dtype=np.uint8)
        cv2.putText(mask, "AB", (45, 155), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 8, cv2.LINE_8)
        candidates = detect_text_regions(source, mask)
        self.assertGreaterEqual(len(candidates), 1)
        for candidate in candidates:
            self.assertGreaterEqual(len(candidate.polygon), 3)
            self.assertTrue(all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in candidate.polygon))
            self.assertTrue(candidate.color.startswith("#"))

    def test_region_mask_selects_only_requested_status(self) -> None:
        pending = ImageLabRegion("区域-1", ((0.1, 0.1), (0.4, 0.1), (0.4, 0.4), (0.1, 0.4)), status="pending")
        processed = ImageLabRegion("区域-2", ((0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)), status="processed")
        mask = region_mask((100, 100), [pending, processed], statuses={"processed"})
        self.assertFalse(mask[20, 20])
        self.assertTrue(mask[75, 75])

    def test_line_candidates_share_one_character_grid_despite_ocr_jitter(self) -> None:
        height, width = 240, 180
        source = np.full((height, width, 3), 238, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        for center_x in (40, 80, 120):
            for top in (12, 52, 92, 132, 172):
                cv2.rectangle(source, (center_x - 11, top + 5), (center_x + 11, top + 31), (35, 35, 35), -1)
                cv2.rectangle(mask, (center_x - 11, top + 5), (center_x + 11, top + 31), 255, -1)

        boxes = np.asarray(
            [
                ((25, 3), (55, 3), (55, 237), (25, 237)),
                ((66, 8), (96, 8), (96, 232), (66, 232)),
                ((105, 1), (135, 1), (135, 239), (105, 239)),
            ],
            dtype=np.float32,
        )

        class Result:
            scores = (0.9, 0.88, 0.91)

            def __init__(self) -> None:
                self.boxes = boxes

        with patch("core.image_regions._rapidocr_engine", return_value=lambda *_args, **_kwargs: Result()):
            candidates = detect_text_regions(source, mask)

        rows_by_column: dict[int, list[float]] = {}
        for candidate in candidates:
            points = np.asarray(candidate.polygon, dtype=np.float32)
            center_x = float(np.mean(points[:, 0]) * width)
            column = int(round(center_x / 40.0))
            rows_by_column.setdefault(column, []).append(float(np.min(points[:, 1]) * height))
        self.assertEqual(set(rows_by_column), {1, 2, 3})
        ordered = [sorted(rows_by_column[column]) for column in (1, 2, 3)]
        common_count = min(len(rows) for rows in ordered)
        self.assertGreaterEqual(common_count, 5)
        for row_index in range(common_count):
            aligned = [rows[row_index] for rows in ordered]
            self.assertLessEqual(max(aligned) - min(aligned), 1.25)

    def test_line_candidates_include_strokes_at_cell_edges(self) -> None:
        height, width = 240, 180
        source = np.full((height, width, 3), 238, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        for center_x in (40, 80, 120):
            for top in (0, 40, 80, 120, 160, 200):
                cv2.rectangle(
                    source,
                    (center_x - 19, top + 1),
                    (center_x + 18, min(height - 1, top + 38)),
                    (35, 35, 35),
                    -1,
                )
                cv2.rectangle(
                    mask,
                    (center_x - 19, top + 1),
                    (center_x + 18, min(height - 1, top + 38)),
                    255,
                    -1,
                )

        boxes = np.asarray(
            [
                ((25, 0), (55, 0), (55, 239), (25, 239)),
                ((66, 0), (96, 0), (96, 239), (66, 239)),
                ((105, 0), (135, 0), (135, 239), (105, 239)),
            ],
            dtype=np.float32,
        )

        class Result:
            scores = (0.9, 0.88, 0.91)

            def __init__(self) -> None:
                self.boxes = boxes

        with patch("core.image_regions._rapidocr_engine", return_value=lambda *_args, **_kwargs: Result()):
            candidates = detect_text_regions(source, mask)

        covered = np.zeros((height, width), dtype=np.uint8)
        for candidate in candidates:
            points = np.rint(np.asarray(candidate.polygon) * (width, height)).astype(np.int32)
            cv2.fillPoly(covered, [points], 1)
        ink = mask > 0
        coverage = float(np.count_nonzero(ink & (covered > 0))) / max(1, int(np.count_nonzero(ink)))
        self.assertGreaterEqual(coverage, 0.995)
        for center_x in (40, 80, 120):
            for top in (0, 40, 80, 120, 160, 200):
                candidate = min(
                    candidates,
                    key=lambda item: abs(
                        float(np.mean(np.asarray(item.polygon)[:, 0])) * width - center_x
                    )
                    + abs(
                        float(np.mean(np.asarray(item.polygon)[:, 1])) * height
                        - (top + 19)
                    ),
                )
                polygon = np.rint(np.asarray(candidate.polygon) * (width, height)).astype(
                    np.int32
                )
                self.assertGreaterEqual(
                    cv2.pointPolygonTest(polygon.astype(np.float32), (center_x - 19, top + 1), False),
                    0.0,
                )
                self.assertGreaterEqual(
                    cv2.pointPolygonTest(
                        polygon.astype(np.float32),
                        (center_x + 18, min(height - 1, top + 38)),
                        False,
                    ),
                    0.0,
                )

    def test_line_candidates_use_padded_irregular_polygons_for_uneven_strokes(self) -> None:
        height, width = 160, 120
        source = np.full((height, width, 3), 240, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        for center_x in (35, 75):
            for top in (5, 45, 85, 125):
                cv2.line(source, (center_x - 15, top + 5), (center_x - 15, min(height - 1, top + 32)), (30, 30, 30), 5)
                cv2.line(source, (center_x - 15, top + 32), (center_x + 14, min(height - 1, top + 32)), (30, 30, 30), 5)
                cv2.line(mask, (center_x - 15, top + 5), (center_x - 15, min(height - 1, top + 32)), 255, 5)
                cv2.line(mask, (center_x - 15, top + 32), (center_x + 14, min(height - 1, top + 32)), 255, 5)

        boxes = np.asarray(
            [
                ((20, 0), (50, 0), (50, 159), (20, 159)),
                ((60, 0), (90, 0), (90, 159), (60, 159)),
            ],
            dtype=np.float32,
        )

        class Result:
            scores = (0.9, 0.9)

            def __init__(self) -> None:
                self.boxes = boxes

        with patch("core.image_regions._rapidocr_engine", return_value=lambda *_args, **_kwargs: Result()):
            candidates = detect_text_regions(source, mask)

        self.assertTrue(any(len(candidate.polygon) > 4 for candidate in candidates))
        candidate = min(
            candidates,
            key=lambda item: abs(float(np.mean(np.asarray(item.polygon)[:, 0])) * width - 35.0)
            + abs(float(np.mean(np.asarray(item.polygon)[:, 1])) * height - 19.0),
        )
        polygon = np.rint(np.asarray(candidate.polygon) * (width, height)).astype(np.int32)
        self.assertGreaterEqual(
            cv2.pointPolygonTest(polygon.astype(np.float32), (20.0, 19.0), False),
            0.0,
        )
        self.assertGreaterEqual(
            cv2.pointPolygonTest(polygon.astype(np.float32), (17.0, 19.0), False),
            0.0,
        )

    def test_unreliable_ocr_boxes_fall_back_to_foreground_components(self) -> None:
        height, width = 180, 260
        source = np.full((height, width, 3), 238, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(source, (90, 100), (130, 140), (35, 35, 35), -1)
        cv2.rectangle(mask, (90, 100), (130, 140), 255, -1)

        # 两个 OCR 框完全落在空白页边，不能用来推断字符网格。
        boxes = np.asarray(
            [
                ((4, 4), (100, 4), (100, 20), (4, 20)),
                ((8, 28), (104, 28), (104, 44), (8, 44)),
            ],
            dtype=np.float32,
        )

        class Result:
            scores = (0.95, 0.94)

            def __init__(self) -> None:
                self.boxes = boxes

        with patch("core.image_regions._rapidocr_engine", return_value=lambda *_args, **_kwargs: Result()):
            candidates = detect_text_regions(source, mask)

        self.assertTrue(candidates)
        centers = [
            np.mean(np.asarray(candidate.polygon), axis=0) * (width, height)
            for candidate in candidates
        ]
        self.assertTrue(any(80.0 <= center[0] <= 140.0 and 90.0 <= center[1] <= 150.0 for center in centers))


if __name__ == "__main__":
    unittest.main()
