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

## Offline mode (tests / development only)

> **Not a user-facing feature.** End users always run with a real
> `OPENAI_API_KEY`; the stub outputs below are not useful results, only a way
> to keep the harness importable and the UI navigable when no key is available.

If `OPENAI_API_KEY` is unset (empty or missing), every LLM-touching node,
extractor, and judge short-circuits to deterministic stub data and the search
clients are skipped — the pipeline runs end-to-end with **zero network calls**.
This exists to keep three workflows hermetic:

- **`pytest`** — `tests/conftest.py` forces `OPENAI_API_KEY=""` so the suite
  runs in ~0.2 s without burning tokens or depending on OpenAI uptime.
- **Fresh clone** — teammates can `pip install -e . && streamlit run …` before
  filling in `.env`; the UI loads, the LangGraph compiles, and the wiring is
  visible (with a yellow "Offline mode" banner up top).
- **UI / wiring iteration** — when changing `streamlit_app.py`, graph topology,
  or `runlog`, you don't need to pay 1-6 minutes per reload to see the result.

When offline, you'll see one stub sub-problem, one stub paper, stub
extractions, and a templated synthesis line — proof the pipeline is connected,
nothing more. Unset the key (`unset OPENAI_API_KEY`) or leave it blank in
`.env` to enter offline mode; set a real key to leave it.
