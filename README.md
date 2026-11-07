# Context-Aware Humor Detection (NLP Mini Project)

## What it does
- Accepts conversational context and a target utterance.
- Trains two comparable TF-IDF + Logistic Regression models:
  1. Baseline: target utterance only.
  2. Context-aware: context + target utterance.
- Evaluates both with accuracy, precision, recall, F1, and confusion matrices.
- Provides an interactive Streamlit demo.

## Important dataset note
`data/humor_context.csv` is a **small illustrative/demo dataset** created to make the project runnable immediately. It is not suitable for claiming generalizable research results. For a proper evaluation, replace it with a larger, human-annotated conversational dataset. A relevant research resource is the M2H2 conversational Hindi humor dataset: https://github.com/declare-lab/M2H2-dataset. Check its license and dataset instructions before use, and map its fields to `context`, `text`, and `label`.

CSV schema:
- `context`: preceding conversation or situation
- `text`: target utterance to classify
- `label`: `humorous` or `non_humorous`

## Setup (Windows PowerShell)
```powershell
cd context_aware_humor_detection
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train.py
.venv\Scripts\python.exe -m streamlit run app.py
```

## Outputs
After `python train.py`:
- `models/baseline.joblib`
- `models/context_aware.joblib`
- `outputs/metrics.json`
- `outputs/classification_report_*.txt`
- `outputs/confusion_matrix_*.png`

## How to prepare real data
1. Obtain a conversational humor dataset and check its license/usage terms.
2. Create a CSV with the three columns above.
3. Ensure labels are consistently `humorous` / `non_humorous` (or `1` / `0`).
4. Replace the demo CSV and run `python train.py` again.
5. Keep context and utterance from the same conversation in the same split if possible, to reduce leakage.
6. Report the actual metrics produced by your run. Do not copy example or demo metrics as final results.

## Limitations
- Humor is subjective, culture-dependent, and difficult to infer from text alone.
- TF-IDF + Logistic Regression is an interpretable baseline, not a deep contextual language model.
- A context-aware feature concatenation is a first step; a transformer fine-tuned on conversational labels could be a future extension.
