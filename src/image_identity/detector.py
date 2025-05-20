import cv2
from typing import List, Tuple


def _filter_candidates(candidates, img_shape):
    """Return only bounding boxes that look like license plates."""
    height, width = img_shape[:2]
    img_area = height * width
    result = []
    for (x, y, w, h) in candidates:
        if h == 0:
            continue
        aspect = w / float(h)
        area = w * h
        if 2.0 <= aspect <= 6.0 and 0.001 * img_area <= area <= 0.1 * img_area:
            result.append((x, y, w, h))
    return result


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
    if hasattr(plates, 'tolist'):
        plates = plates.tolist()
    plates = _filter_candidates(plates, image.shape)
    return plates
