import unittest


try:
    import numpy as np
    import cv2
    from image_identity import processor
    blur_region = processor.blur_region
    overlay_text = processor.overlay_text
    overlay_image = processor.overlay_image
    overlay_image = processor.overlay_image
    _deps_available = True
except Exception:  # pragma: no cover - dependencies may be missing
    np = None
    cv2 = None
    blur_region = None
    overlay_text = None
    _deps_available = False

@unittest.skipUnless(_deps_available, "NumPy and OpenCV are required")
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

    def test_overlay_image_rotated(self):
        img = self.image.copy()
        overlay = np.ones((20, 40, 3), dtype=np.uint8) * 255
        # draw a rotated rectangle inside bbox to provide orientation cues
        pts = np.array([[60, 40], [100, 40], [100, 60], [60, 60]], dtype="float32")
        M = cv2.getRotationMatrix2D((80, 50), 30, 1)
        rpts = cv2.transform(np.array([pts]), M)[0].astype(int)
        cv2.fillConvexPoly(img, rpts, (255, 255, 255))
        overlay_image(img, self.bbox, overlay)
        self.assertTrue(np.any(img != 0))

    def test_overlay_image_keeps_size(self):
        img = self.image.copy()
        overlay = np.ones((20, 40, 3), dtype=np.uint8) * 255
        overlay_image(img, self.bbox, overlay)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        x, y, w, h = self.bbox
        bbox_mask = np.zeros_like(mask)
        cv2.rectangle(bbox_mask, (x, y), (x + w, y + h), 255, -1)
        outside = cv2.bitwise_and(mask, cv2.bitwise_not(bbox_mask))
        self.assertLess(np.sum(outside), 0.2 * w * h)


if __name__ == '__main__':
    unittest.main()
