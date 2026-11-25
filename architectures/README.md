# Architectures from scratch — model code

The from-scratch models behind `/models/*` on the site. Each model folder holds:

- `<model>.py` — the model, written to be read top-down; the page shows this exact source
- `load_hf_weights.py` — copies the official released weights into it
- `test_<model>.py` — checks it against the Hugging Face reference
- `export_site_data.py` — runs one prompt and writes `src/data/models/<model>.json` (code snippets + real tensors)

The site never runs Python: the exported JSON is committed and the Astro build reads it.

## Setup (once)

Torch comes from the system/conda install; the venv adds the rest.

```bash
cd architectures
uv venv --python "$(which python3)" --system-site-packages .venv
uv pip install --python .venv/bin/python "transformers>=4.45,<5" tiktoken pytest
```

`transformers` 5.x needs torch ≥ 2.6; with torch 2.5 keep the `<5` pin.

## After changing a model

```bash
cd architectures/gpt2
../.venv/bin/python -m pytest -q          # still matches Hugging Face?
../.venv/bin/python export_site_data.py   # refresh the page's code and tensors
```

Then `npm run build` from the repo root.
