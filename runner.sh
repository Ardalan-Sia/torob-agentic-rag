cat > run_phase1.sh << 'BASH'
#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/torob_rag/src"

echo "=== Phase 1 Runner ==="
echo "PYTHONPATH=$PYTHONPATH"
echo

# --- EDA & Cleaning ---
python torob_rag/src/torob_rag/eda/checkpoint_01_load_sanity.py
python torob_rag/src/torob_rag/eda/checkpoint_02_eda_report.py
python torob_rag/src/torob_rag/cleaning/checkpoint_03_clean_unify.py

# --- Subset (logs-weighted) ---
python torob_rag/src/torob_rag/eda/checkpoint_04a2_build_subset_from_searches.py

# --- Text embeddings + FAISS (your existing text pipeline) ---
# NOTE: If your text-embedding checkpoint filename differs, edit the next line.
python torob_rag/src/torob_rag/embeddings/checkpoint_04b_text_embeddings.py
python torob_rag/src/torob_rag/vector_db/checkpoint_05a_build_faiss_text.py

# --- Text eval + error analysis ---
python torob_rag/src/torob_rag/eval/checkpoint_06_text_only_eval.py
python torob_rag/src/torob_rag/eval/checkpoint_07a_make_failures_readable.py

# --- Image-only baseline (subset->download->embed->faiss->sanity) ---
python torob_rag/src/torob_rag/image/checkpoint_08a_build_image_subset.py
python torob_rag/src/torob_rag/image/checkpoint_08b_download_images.py
python torob_rag/src/torob_rag/image/checkpoint_08c_clip_image_embeddings.py
python torob_rag/src/torob_rag/image/checkpoint_08d_build_faiss_image.py
python torob_rag/src/torob_rag/image/checkpoint_08f_image_index_sanity.py

# --- Multimodal late fusion (prepare corpus->text embed->faiss->sanity) ---
python torob_rag/src/torob_rag/multimodal/checkpoint_09a_prepare_mm_corpus.py
python torob_rag/src/torob_rag/multimodal/checkpoint_09b_embed_mm_text.py
python torob_rag/src/torob_rag/multimodal/checkpoint_09c_build_faiss_mm_text.py
python torob_rag/src/torob_rag/multimodal/checkpoint_09e_multimodal_sanity_eval.py

echo
echo "✅ Phase 1 completed. Key artifacts:"
echo "- torob_rag/artifacts/eval/text/eval_summary.json"
echo "- torob_rag/artifacts/eval/text/failure_examples_readable.csv"
echo "- torob_rag/artifacts/eval/image/clip_v1/sanity_report.json"
echo "- torob_rag/artifacts/eval/multimodal/clip_v1/mm_sanity_eval.json"
BASH

chmod +x run_phase1.sh
