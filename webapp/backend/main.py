from __future__ import annotations

from .app import create_app

app = create_app()


def run_dev_server() -> None:
    import uvicorn

    uvicorn.run("webapp.backend.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run_dev_server()
