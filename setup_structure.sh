#!/usr/bin/env bash
set -e

# Run this from inside your insight-hub folder (where pyproject.toml already lives)

# 1. Create the full folder structure
mkdir -p data/raw data/processed
mkdir -p src/ingestion src/features src/models src/api src/rag
mkdir -p notebooks tests docker .github/workflows

# 2. Keep empty folders tracked by git
touch data/raw/.gitkeep data/processed/.gitkeep
touch src/ingestion/.gitkeep src/features/.gitkeep src/models/.gitkeep src/api/.gitkeep src/rag/.gitkeep
touch notebooks/.gitkeep tests/.gitkeep docker/.gitkeep

# 3. Make src/ a proper importable Python package
touch src/__init__.py src/ingestion/__init__.py src/features/__init__.py \
      src/models/__init__.py src/api/__init__.py src/rag/__init__.py

# 4. Env template (pyproject.toml, .gitignore, .python-version, README.md already exist from uv init)
touch .env.example

echo "Folder structure created inside insight-hub."
echo "Dependencies now go through: uv add <package>  (not requirements.txt)"
