"""Loading and encoding local images for the vision API.

Isolates all the file + format concern in one place: read the bytes,
base64-encode them, and map the file extension to the `media_type` the Messages
API expects. Friendly errors for the three ways a learner-supplied path goes
wrong: missing file, unsupported format, or too large.
"""

import base64
from pathlib import Path

# Formats the Claude vision API accepts. (Animations use only the first frame.)
SUPPORTED_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# The direct Claude API accepts up to 10 MB of base64-encoded image data.
MAX_BASE64_BYTES = 10 * 1024 * 1024


def load_image(path: str) -> tuple[str, str]:
    """Read a local image and return (base64_data, media_type).

    Raises a clear error for each failure mode so the CLI can show the learner a
    helpful message instead of a stack trace:
      - unsupported extension -> ValueError
      - missing file          -> FileNotFoundError
      - too large             -> ValueError
    """
    image_path = Path(path).expanduser()

    media_type = SUPPORTED_MEDIA_TYPES.get(image_path.suffix.lower())
    if media_type is None:
        supported = ", ".join(sorted(SUPPORTED_MEDIA_TYPES))
        raise ValueError(f"Unsupported image type {image_path.suffix!r}. Use one of: {supported}.")

    try:
        raw = image_path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No image found at {path!r}.") from exc

    encoded = base64.standard_b64encode(raw).decode("utf-8")
    if len(encoded) > MAX_BASE64_BYTES:
        size_mb = len(encoded) // (1024 * 1024)
        raise ValueError(f"Image is too large ({size_mb} MB encoded); the maximum is 10 MB.")

    return encoded, media_type
