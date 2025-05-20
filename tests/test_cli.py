import os
import unittest

try:
    import cv2
    from image_identity.cli import process_image
    _deps_available = True
except Exception:
    cv2 = None
    process_image = None
    _deps_available = False

@unittest.skipUnless(_deps_available, "OpenCV is required")
class TestCLI(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test-data'))
        self.images = [
            'front.jpg',
            'back.jpg',
            'front_angle.jpg',
            'back_angle.jpg'
        ]
        self.output_dir = os.path.join(self.data_dir, 'out')
        os.makedirs(self.output_dir, exist_ok=True)

    def test_process_sample_images(self):
        missing = [img for img in self.images if not os.path.exists(os.path.join(self.data_dir, img))]
        if missing:
            self.skipTest(f"Missing sample images: {missing}")
        for name in self.images:
            in_path = os.path.join(self.data_dir, name)
            process_image(in_path, self.output_dir, text='TEST')
            out_path = os.path.join(self.output_dir, name)
            self.assertTrue(os.path.exists(out_path))

if __name__ == '__main__':
    unittest.main()
