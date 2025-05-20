from pathlib import Path

# Ensure the package works when running from the repository root without
# installation by adding the "src" directory to the package search path.
_src = Path(__file__).resolve().parent.parent / "src" / "image_identity"
if _src.exists():
    __path__.append(str(_src))
