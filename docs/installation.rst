Installation
============

Basic install
-------------

Install from PyPI:

.. code-block:: bash

   pip install jacscanomaly

The default PSPL survey workflow uses the compiled C++ backend. A normal
source install therefore builds ``jacscanomaly._cpp_grid`` during installation.
If you need to force a local source build instead of using an already-built
wheel, run:

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

The package includes a C++ grid backend and C++ PSPL fitting backend. They are
built through ``setup.py`` using OpenMP:

.. code-block:: bash

   pip install -e .

The C++ backend is required by the default PSPL survey workflow. If the
extension cannot be built, installation should fail rather than producing a
runtime-only failure when ``FinderConfig(grid_backend="cpp")`` is used.

If the extension does not build, check that your compiler supports ``C++17``
and OpenMP. On Linux this usually means installing a recent ``gcc``/``g++``
toolchain. On macOS this may require installing ``libomp`` and using compiler
flags that can find it.

If you only need a temporary pure-Python/JAX workaround for debugging, request
the JAX backend explicitly in your configuration:

.. code-block:: python

   config = FinderConfig(
       grid_backend="jax",
       single_fit_backend="jax",
   )
