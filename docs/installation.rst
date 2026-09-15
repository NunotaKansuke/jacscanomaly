Installation
============

Basic install
-------------

Install from PyPI:

.. code-block:: bash

   pip install jacscanomaly

The default PSPL survey workflow uses the compiled C++ backend. PyPI provides
platform wheels containing this backend, so a normal install does not require
a local compiler. If no compatible wheel is available for your platform,
``pip`` falls back to a source build. To force that source build explicitly,
run:

.. code-block:: bash

   pip install --no-binary jacscanomaly jacscanomaly

You can verify that the compiled backend is available with:

.. code-block:: bash

   python -c "import jacscanomaly._cpp_grid"

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

FSPL magnification
------------------

FSPL fitting uses the compiled finite-source magnification backend and does
not require ``microjax``. The package still uses JAX for the optional anomaly
grid and legacy diagnostic helpers.

C++ backend
-----------

The package includes a C++ anomaly-grid backend and compiled magnification /
parallax evaluators. They are built through ``setup.py`` using OpenMP:

.. code-block:: bash

   pip install -e .

The compiled backend is required when ``grid_backend="cpp"`` or an FSPL /
parallax fitter is used. If the extension cannot be built, installation
should fail rather than producing a runtime-only failure.

If the extension does not build, check that your compiler supports ``C++17``
and OpenMP. On Linux this usually means installing a recent ``gcc``/``g++``
toolchain. On macOS this may require installing ``libomp`` and using compiler
flags that can find it.

For a temporary JAX anomaly-grid comparison, use:

.. code-block:: python

   config = FinderConfig(
       grid_backend="jax",
   )
