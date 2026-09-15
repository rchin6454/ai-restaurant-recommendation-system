# AI-Powered Restaurant Recommendation System

Restaurant recommendations over the Zomato Bangalore dataset, filtered deterministically and ranked/explained by Claude. See [docs/architecture.md](docs/architecture.md) and [docs/implementation-plan.md](docs/implementation-plan.md).

## Setup

Requires Python 3.11+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add ANTHROPIC_API_KEY; empty is allowed (degraded mode)
```

If `pip` crashes with `invalid literal for int() with base 10: ''` (a Homebrew Python whose `pyexpat` can't load against the system libexpat), use a uv-managed interpreter instead:

```bash
brew install uv
uv venv --python 3.12 --python-preference only-managed .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Tests

Run from the repository root:

```bash
pytest
```

## Configuration

All tunables live in [src/config.py](src/config.py) and can be overridden by environment variables or `.env` (see [.env.example](.env.example)). Relative paths such as `CATALOG_PATH` resolve against the repository root.
