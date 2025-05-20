# Image Identity

This project provides utilities to detect license plates in vehicle images and
remove or replace them with custom text or an overlay image. The main entry
point is a command line interface that processes one or more images.

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Tesseract OCR

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```
Additionally, Tesseract must be installed on your system. On Debian-based
systems you can install it via:

```bash
sudo apt-get install tesseract-ocr
```

Alternatively, you can run the provided setup script to install these
dependencies automatically:

```bash
./setup_requirements.sh
```
The script installs the `tesseract-ocr` system package and then runs
`pip install -r requirements.txt`.

## Usage


Example images are stored in the `test-data` directory. The CLI can be invoked
as follows:

```bash
python -m image_identity.cli \
    --input "test-data/JAECOO_front.jpg" \
    --input "test-data/JAECOO rear.jpg" \
    --input "test-data/JAECOO front 3 quarter.jpg" \
    --input "test-data/JAECOO rear 3 quarter.jpg" \
    --output processed_images --text "DEMO"
```

You can also provide an overlay image for the plate using `--overlay` instead of
`--text`.
To automatically recognise the existing plate text and replace it with custom text
use the `--replace-with` option:

```bash
python -m image_identity.cli \
    --input "test-data/JAECOO_front.jpg" --output processed_images \
    --replace-with "ANON"
```

Processed images are written to the specified output directory with license
plates blurred, replaced by text, or overlaid with a custom image.
