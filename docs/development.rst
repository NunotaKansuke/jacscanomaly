Development
===========

This page describes the checks expected before opening a pull request.

Editable install
----------------

.. code-block:: bash

   pip install -e ".[dev]"

Unit tests
----------

.. code-block:: bash

   pytest

The current unit tests cover fast deterministic pieces of the package. New
tests should be added with new behavior, especially for:

* candidate selection criteria,
* result serialization or summaries,
* season splitting and grid construction,
* small fixed light-curve examples.

Coverage
--------

.. code-block:: bash

   coverage run -m pytest
   coverage report

Coverage is currently a diagnostic, not a hard gate. It should be used to show
where the test suite is thin before CI requirements become stricter.

Documentation
-------------

Build docs locally with warnings treated as errors:

.. code-block:: bash

   sphinx-build -W -b html docs docs/_build/html

Generated files under ``docs/_build`` and ``docs/generated`` are not committed.

Continuous integration
----------------------

The GitHub Actions workflow runs:

* package install with ``.[test,docs]``,
* ``pytest``,
* ``coverage report``,
* ``sphinx-build -W``.

This keeps both the test suite and the ReadTheDocs source buildable.
