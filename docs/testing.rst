Testing
=======

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
