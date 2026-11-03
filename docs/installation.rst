Installation
============

.. code-block:: bash

   pip install jacscanomaly

The wheels contain the compiled backends (scan grid, FSPL magnification,
parallax). Without a wheel for your platform, ``pip`` builds from source; that
needs a C++17 compiler with OpenMP (on macOS, ``libomp``).

From the repository:

.. code-block:: bash

   git clone git@github.com:NunotaKansuke/jacscanomaly.git
   cd jacscanomaly
   pip install -e ".[dev]"

Tests:

.. code-block:: bash

   pytest
