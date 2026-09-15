"""AI-powered restaurant recommendation system."""

import sys

# Fail fast with a setup message instead of a SyntaxError deep in a later phase (O-09).
if sys.version_info < (3, 11):
    raise RuntimeError(
        f"Python 3.11+ is required (found {sys.version.split()[0]}). "
        "Create the virtualenv with a newer interpreter, e.g. `python3.12 -m venv .venv`."
    )
