import unittest
import numpy as np
import cv2

from image_identity.processor import blur_region, overlay_text


class TestProcessor(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 200, 3), dtype=np.uint8)
        self.bbox = (50, 25, 100, 50)

    def test_blur_region(self):
        img = self.image.copy()
        img[25:75, 50:150] = 255
        blur_region(img, self.bbox)
        region = img[25:75, 50:150]
        self.assertFalse(np.all(region == 255))

    def test_overlay_text(self):
        img = self.image.copy()
        overlay_text(img, self.bbox, "TEST")
        # Expect some non-zero pixels due to text drawing
        region = img[25:75, 50:150]
        self.assertTrue(np.any(region != 0))


if __name__ == '__main__':
    unittest.main()
