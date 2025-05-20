import cv2
from typing import List, Tuple


def detect_license_plates(image) -> List[Tuple[int, int, int, int]]:
    """Detect license plates in an image.

    Args:
        image: BGR image loaded via OpenCV.

    Returns:
        A list of bounding boxes (x, y, w, h) for detected plates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    return plates.tolist() if hasattr(plates, 'tolist') else []
