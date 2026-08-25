# Dockerfile
#
# Builds a container image for the InsightHub API.
# Using uv inside Docker for fast, reproducible dependency installs.

FROM python:3.12-slim

# Install uv (the same tool we've been using locally)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (not the whole project yet).
# Docker caches layers — if only your source code changes but not
# dependencies, this layer gets reused instead of reinstalling everything,
# making rebuilds much faster.
COPY pyproject.toml uv.lock ./

# Install dependencies into the image (not a local .venv this time)
RUN uv sync --frozen --no-dev

# Now copy the rest of the project (source code, trained models, etc.)
COPY . .

# Tell Docker this container listens on port 8000
EXPOSE 8000

# The command that runs when the container starts
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
