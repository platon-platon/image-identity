import os
import unittest

try:
    import cv2
    from image_identity.ocr import recognize_text
    from image_identity.detector import detect_license_plates
    _deps_available = True
except Exception:
    cv2 = None
    recognize_text = None
    detect_license_plates = None
    _deps_available = False


def _load_sample_image():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test-data'))
    for name in os.listdir(data_dir):
        if name.endswith('.jpg'):
            path = os.path.join(data_dir, name)
            img = cv2.imread(path)
            if img is not None:
                return img
    return None


@unittest.skipUnless(_deps_available, "OpenCV and pytesseract are required")
class TestOCR(unittest.TestCase):
    def setUp(self):
        self.image = _load_sample_image()
        if self.image is None:
            self.skipTest("No sample image available")

    def test_recognize_text_returns_string(self):
        plates = detect_license_plates(self.image)
        if not plates:
            self.skipTest("No license plates detected")
        text = recognize_text(self.image, plates[0])
        self.assertIsInstance(text, str)


if __name__ == '__main__':
    unittest.main()

