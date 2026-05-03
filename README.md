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
harness can be exercised without network access. This is the default for tests.

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
app/streamlit_app.py     Web UI entry
scripts/                 CLI runners (run_agent, run_eval)
tests/                   pytest suite
data/                    cache + eval datasets (gitignored content)
notebooks/               sandbox notebooks
```

## Make targets

```bash
make env       # conda env create -f environment.yml
make run       # streamlit run app/streamlit_app.py
make test      # pytest -q
make eval      # run evaluation harness on a results JSONL
make lint      # ruff + black --check
```

## Task division (per proposal)

- Lawrence — agent framework + backend pipeline
- Eric — academic search + paper parsing + web interface
- Haoyang — summary / flowchart generation, prompt design
- Howard — evaluation (LLM-as-judge + human assessment)
