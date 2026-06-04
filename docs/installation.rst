Installation
============

Basic install
-------------

Install from PyPI:

.. code-block:: bash

   pip install jacscanomaly

For local development, install the repository in editable mode:

.. code-block:: bash

   git clone git@github.com:NunotaKansuke/jacscanomaly.git
   cd jacscanomaly
   pip install -e .

Optional dependencies
---------------------

Documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

Test dependencies:

.. code-block:: bash

   pip install -e ".[test]"

All development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

FSPL and ``microjax``
------------------------

The PSPL workflow only needs the standard package dependencies. FSPL
magnification currently relies on the GitHub source version of ``microjax``
because the PyPI package may not expose ``microjax.fastlens.fspl_disk``.

Install ``microjax`` from source before using FSPL fitters:

.. code-block:: bash

   git clone https://github.com/ShotaMiyazaki94/microjax.git
   cd microjax
   pip install -e .

C++ backend
-----------

The package includes an experimental C++ grid backend and C++ PSPL fitting
backend. They are built through ``setup.py`` using OpenMP:

.. code-block:: bash

   pip install -e .

If the extension does not build, first check that your compiler supports
``-fopenmp`` and C++17.
