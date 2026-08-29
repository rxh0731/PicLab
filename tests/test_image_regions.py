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


if __name__ == "__main__":
    unittest.main()
