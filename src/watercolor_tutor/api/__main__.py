"""Run the API with `python -m watercolor_tutor.api` (a thin uvicorn launcher).

Equivalent to `uvicorn watercolor_tutor.api:app --reload`; provided so the web
entry point is as discoverable as the CLI (`python -m watercolor_tutor`).
"""

import uvicorn


def main() -> None:
    """Serve the API locally with autoreload."""
    uvicorn.run("watercolor_tutor.api:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
