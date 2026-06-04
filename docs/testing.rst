Testing
=======

``jacscanomaly`` uses ``pytest`` for unit tests and ``coverage`` for coverage
measurement. The initial tests focus on deterministic core components:

- season splitting,
- candidate quality criteria,
- result summaries,
- cluster extraction,
- weighted linear photometry utilities.

Install the test dependencies with:

.. code-block:: bash

   pip install -e ".[test]"

Run the unit tests:

.. code-block:: bash

   pytest

Run the tests with coverage:

.. code-block:: bash

   coverage run -m pytest
   coverage report

Build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs docs/_build/html

Current scope
-------------

The first test suite intentionally avoids full survey-scale fits. Those tests
are more expensive and should be added later as integration tests with small
fixed light curves. Unit tests should stay fast enough to run on every commit.
