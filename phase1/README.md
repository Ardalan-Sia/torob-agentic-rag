# Torob RAG - Phase 1

## Documentation

The Phase 1 Persian LaTeX report is available here:

```text
docs/phase1_report_fa.tex
docs/phase1_report_fa.pdf
```

## Presentation Notebook

The read-only presentation notebook summarizes the complete Phase 1 pipeline and reads the generated artifacts without modifying the raw data:

```text
torob_rag/notebooks/Torob_Phase1_Pipeline_Presentation.ipynb
```

## Setup
Open this folder in VSCode:

```powershell
<PROJECT_ROOT>\phase1
```

Then run:

```powershell
cd "<PROJECT_ROOT>\phase1"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = ".\torob_rag\src"
```
