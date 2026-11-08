FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY sanad ./sanad
RUN pip install --no-cache-dir ".[api,pdf]"

COPY data ./data
COPY eval ./eval

# The index is built on first request (or run `python -m sanad index` in the container),
# so the image works both offline and with a GEMINI_API_KEY supplied at runtime.
# Hosting platforms (Render, Railway, Fly) pass the port in $PORT.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "uvicorn sanad.api:app --host 0.0.0.0 --port ${PORT}"]
