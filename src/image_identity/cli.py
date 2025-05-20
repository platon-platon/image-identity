import argparse
import cv2
import os

from .detector import detect_license_plates
from .processor import blur_region, overlay_text, overlay_image


def process_image(path: str, output_dir: str, text: str = None, overlay_path: str = None):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    plates = detect_license_plates(image)
    overlay_img = None
    if overlay_path:
        overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay_img is None:
            raise FileNotFoundError(f"Unable to read overlay: {overlay_path}")

    for bbox in plates:
        if overlay_img is not None:
            overlay_image(image, bbox, overlay_img)
        elif text is not None:
            overlay_text(image, bbox, text)
        else:
            blur_region(image, bbox)

    os.makedirs(output_dir, exist_ok=True)
    name = os.path.basename(path)
    out_path = os.path.join(output_dir, name)
    cv2.imwrite(out_path, image)
    print(f"Processed image saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Process images to anonymize license plates.")
    parser.add_argument('--input', '-i', action='append', required=True, help='Input image path')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--text', help='Custom text to overlay on license plates')
    parser.add_argument('--overlay', help='Path to custom plate image overlay')
    args = parser.parse_args()

    for img_path in args.input:
        process_image(img_path, args.output, text=args.text, overlay_path=args.overlay)


if __name__ == '__main__':
    main()
