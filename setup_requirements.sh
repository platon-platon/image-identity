#!/bin/bash
set -e
apt-get update
apt-get install -y tesseract-ocr
pip install -r requirements.txt
