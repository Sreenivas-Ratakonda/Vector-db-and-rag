#!/usr/bin/env bash
# One-command setup for macOS / Linux:  ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=$(command -v python3.12 || command -v python3.11 || command -v python3 || true)
[ -z "$PY" ] && { echo "Python 3 not found"; exit 1; }
echo "Using: $($PY --version)"

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.txt --quiet

echo "Downloading the embedding model (one time, ~80 MB)..."
.venv/bin/python -c "import sys; sys.path.insert(0, 'rag'); from common import get_embedder; get_embedder()"

cat <<'EOF'

Setup complete. Next:
  source .venv/bin/activate
  python rag/step1_embeddings.py
  python rag/step2_rag_without_vectordb.py
  python rag/step3_rag_with_chromadb.py
EOF
