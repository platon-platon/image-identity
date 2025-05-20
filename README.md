# Image Identity

This project provides utilities to detect license plates in vehicle images and
remove or replace them with custom text or an overlay image. The main entry
point is a command line interface that processes one or more images.

## Requirements

- Python 3.8+
- OpenCV
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m image_identity.cli \
    --input front.jpg --input back.jpg --input front_angle.jpg --input back_angle.jpg \
    --output processed_images --text "DEMO"
```

You can also provide an overlay image for the plate using `--overlay` instead of
`--text`.

Processed images are written to the specified output directory with license
plates blurred, replaced by text, or overlaid with a custom image.
