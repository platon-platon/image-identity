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


def _estimate_rotated_rect(roi):
    """Return cv2.minAreaRect for the prominent contour in ``roi`` or ``None``."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return cv2.minAreaRect(contour)


def overlay_image(image, bbox: Tuple[int, int, int, int], overlay) -> None:
    """Overlay another image respecting plate angle when possible."""
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    rect = _estimate_rotated_rect(roi)

    if rect and rect[1][0] > 1 and rect[1][1] > 1:
        # Convert rect centre to absolute image coordinates
        center = (rect[0][0] + x, rect[0][1] + y)
        size = rect[1]
        box = cv2.boxPoints((center, size, rect[2])).astype("float32")
        ow, oh = int(size[0]), int(size[1])
    else:
        box = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype="float32",
        )
        ow, oh = w, h

    overlay_resized = cv2.resize(overlay, (ow, oh))
    src_pts = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, box)
    warped = cv2.warpPerspective(
        overlay_resized, M, (image.shape[1], image.shape[0]), borderValue=(0, 0, 0, 0)
    )

    if warped.shape[2] == 4:
        mask = warped[:, :, 3]
        overlay_rgb = warped[:, :, :3]
    else:
        overlay_rgb = warped
        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)
    for c in range(3):
        bg = cv2.bitwise_and(image[:, :, c], image[:, :, c], mask=mask_inv)
        fg = cv2.bitwise_and(overlay_rgb[:, :, c], overlay_rgb[:, :, c], mask=mask)
        image[:, :, c] = cv2.add(bg, fg)
