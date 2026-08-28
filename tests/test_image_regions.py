"""图片实验室文字区域检测测试。"""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
