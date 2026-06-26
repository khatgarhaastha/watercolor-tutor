"""Tests for image loading + encoding, including the three failure modes."""

import base64
from pathlib import Path

import pytest

from watercolor_tutor.images import load_image

# A real, minimal 1x1 PNG so load_image exercises actual bytes.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def test_load_image_returns_b64_and_media_type(tmp_path: Path) -> None:
    image = tmp_path / "wash.png"
    image.write_bytes(PNG_1X1)

    b64, media_type = load_image(str(image))

    assert media_type == "image/png"
    assert base64.b64decode(b64) == PNG_1X1


def test_load_image_maps_jpg_to_jpeg(tmp_path: Path) -> None:
    image = tmp_path / "wash.jpg"
    image.write_bytes(PNG_1X1)  # bytes don't matter for the media-type mapping
    _, media_type = load_image(str(image))
    assert media_type == "image/jpeg"


def test_load_image_unsupported_type(tmp_path: Path) -> None:
    image = tmp_path / "wash.bmp"
    image.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported image type"):
        load_image(str(image))


def test_load_image_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No image found"):
        load_image(str(tmp_path / "nope.png"))


def test_load_image_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "big.png"
    image.write_bytes(PNG_1X1)
    monkeypatch.setattr("watercolor_tutor.images.MAX_BASE64_BYTES", 1)  # force the limit
    with pytest.raises(ValueError, match="too large"):
        load_image(str(image))
