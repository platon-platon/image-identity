import cv2
import numpy as np
from typing import Tuple


def blur_region(image, bbox: Tuple[int, int, int, int]) -> None:
    """Blur a region in-place defined by bbox in the image."""
    x, y, w, h = bbox
    # Blur the entire image to ensure edge pixels are mixed, then copy the
    # corresponding region back. This avoids unchanged regions when the source
    # area is a solid color.
    blurred_full = cv2.GaussianBlur(image, (51, 51), 0)
    image[y:y+h, x:x+w] = blurred_full[y:y+h, x:x+w]


def overlay_text(image, bbox: Tuple[int, int, int, int], text: str) -> None:
    """Overlay custom text in the bbox region."""
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), thickness=-1)
    font_scale = max(w, h) / 200
    thickness = 2
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = x + (w - size[0]) // 2
    text_y = y + (h + size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def overlay_image(image, bbox: Tuple[int, int, int, int], overlay) -> None:
    """Overlay another image resized to bbox."""
    x, y, w, h = bbox
    overlay_resized = cv2.resize(overlay, (w, h))
    if overlay_resized.shape[2] == 4:
        alpha = overlay_resized[:, :, 3] / 255.0
        for c in range(3):
            image[y:y+h, x:x+w, c] = (1 - alpha) * image[y:y+h, x:x+w, c] + \
                alpha * overlay_resized[:, :, c]
    else:
        image[y:y+h, x:x+w] = overlay_resized
