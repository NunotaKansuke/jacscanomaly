# jacscanomaly documentation

This directory contains the Sphinx source files for the `jacscanomaly`
documentation.

If you are browsing this repository on GitHub, start here. The `.rst` files in
this directory are source files used by Sphinx, so they may show Sphinx syntax
such as `.. toctree::` or `:ref:` when opened directly. The rendered
documentation is produced by building these files with Sphinx or by publishing
them through ReadTheDocs.

## Build locally

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Build the HTML documentation:

```bash
sphinx-build -W -b html docs docs/_build/html
```

Open the generated site:

```bash
python -m webbrowser docs/_build/html/index.html
```

## Documentation pages

- `installation.rst`: installation, development extras, FSPL/microjax notes,
  and C++ backend requirements
- `quickstart.rst`: first full anomaly-search example
- `examples.rst`: copyable examples for fitting, candidate criteria, and
  plotting
- `method.rst`: scan statistic, score, `n_eff`, quality diagnostics, and
  backends
- `configuration.rst`: `FinderConfig` options and how to tune them
- `results.rst`: how to inspect `AnomalyResult` and candidate diagnostics
- `api.rst`: API reference generated from package docstrings
- `testing.rst`: test and coverage commands
- `development.rst`: expected checks before pushing or opening a PR

## ReadTheDocs

The repository includes `.readthedocs.yaml`. Once connected to ReadTheDocs, it
will build the same Sphinx source and publish the rendered HTML site.

