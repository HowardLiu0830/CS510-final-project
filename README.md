# Research Trail Builder

A graph-based knowledge map for navigating scientific literature.

CS510 Group Project — Lawrence Wang (lw41), Eric Chen (ericzc2), Haoyang Wang (hw86), Howard Liu (yl140).

## Quick start (conda)

```bash
# 1. Create env (Python 3.11 + all deps)
conda env create -f environment.yml

# 2. Activate
conda activate cs510

# 3. Configure secrets
cp .env.example .env
# Edit .env: set OPENAI_API_KEY at minimum

# 4. Editable install of the package
pip install -e .

# 5. Smoke test
python -c "import research_trail; from research_trail.agents.graph import compile_graph; compile_graph(); print('OK')"
pytest -q
streamlit run app/streamlit_app.py
```

If `conda env create` fails on pip resolution, fall back to:

```bash
conda create -n cs510 python=3.11 -y
conda activate cs510
pip install -r requirements.txt
pip install -e .
```

## Offline mode

If `OPENAI_API_KEY` is unset, agent nodes return deterministic stub data so the
harness can be exercised without network access. This is the default for tests
(forced in `tests/conftest.py`).

## Layout

```
src/research_trail/      Python package
  config.py              pydantic-settings, loads .env
  llm/                   OpenAI client factory
  agents/                LangGraph state + nodes + compiled graph
  search/                OpenAlex / Semantic Scholar / arXiv clients
  extraction/            LLM-grounded claim/method extraction
  graph/                 concept-graph construction (networkx)
  evaluation/            LLM-as-judge + human eval rubrics
  runlog.py              per-run artifact directory writer
app/streamlit_app.py     Web UI entry
scripts/                 thin CLI wrappers around research_trail.cli
tests/                   pytest suite
data/
  cache/                 search-client cache (gitignored)
  eval/                  eval datasets + aggregated outputs
  runs/                  per-run artifacts: data/runs/<ts>__<slug>/state.json
notebooks/               sandbox notebooks (00_smoke_test.ipynb)
```

## Console scripts

`pip install -e .` registers four entry points (defined in `pyproject.toml`):

| Command                            | Purpose                                                             |
| ---------------------------------- | ------------------------------------------------------------------- |
| `research-trail-agent`             | Run the LangGraph pipeline on a query; writes `data/runs/<ts>__<slug>/`. |
| `research-trail-runs-to-jsonl`     | Collect healthy run states into a JSONL of `{query, output}` records.    |
| `research-trail-eval`              | Score a results JSONL with the LLM judge (and optional human forms).     |
| `research-trail-scaffold-human-eval` | Generate one blank human-eval form per record for reviewers to fill in. |

The `scripts/*.py` files are thin wrappers around the same entry points.

## End-to-end evaluation workflow

```bash
# 1. Generate runs (one query at a time; each writes data/runs/<ts>__<slug>/)
research-trail-agent "graph neural networks for drug discovery"
research-trail-agent "retrieval-augmented generation for scientific QA"

# 2. Collect healthy runs into a JSONL the judge can consume
research-trail-runs-to-jsonl --output data/eval/results.jsonl

# 3. (Optional) scaffold human-eval forms for reviewers
research-trail-scaffold-human-eval \
  --input data/eval/results.jsonl \
  --out-dir data/eval/human_forms/

# 4. Score with the LLM judge, optionally merging filled human forms
research-trail-eval \
  --input data/eval/results.jsonl \
  --output data/eval/runs/latest.json \
  --human-forms data/eval/human_forms/   # optional
```

`run_agent` exits non-zero if the pipeline produced empty sub-problems / papers
/ summary, so empty runs surface immediately. `runs_to_jsonl` skips those by
default (override with `--include-empty`).

## Make targets

```bash
make env       # conda env create -f environment.yml
make run       # streamlit run app/streamlit_app.py
make test      # pytest -q
make eval      # run evaluation harness on a results JSONL
make lint      # ruff + black --check
make format    # ruff --fix + black
make clean     # remove build/ dist/ caches
```

## Task division (per proposal)

- Lawrence — agent framework + backend pipeline
- Eric — academic search + paper parsing + web interface
- Haoyang — summary / flowchart generation, prompt design
- Howard — evaluation (LLM-as-judge + human assessment)
