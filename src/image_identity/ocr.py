import cv2
from typing import Tuple

try:
    import pytesseract
    _ocr_available = True
except Exception:  # pragma: no cover - pytesseract may not be installed
    pytesseract = None
    _ocr_available = False


def recognize_text(image, bbox: Tuple[int, int, int, int]) -> str:
    """Recognize text within the specified bounding box.

    Parameters
    ----------
    image : ndarray
        Source BGR image.
    bbox : tuple of int
        Bounding box ``(x, y, w, h)`` of the region containing text.

    Returns
    -------
    str
        Recognized text or an empty string if OCR is unavailable.
    """
    if not _ocr_available:
        return ""

    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, config="--psm 7")
    return text.strip()

