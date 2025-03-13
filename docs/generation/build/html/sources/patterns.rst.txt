========
Patterns
========

All missingness patterns developed in ImputeGAP are available in the ``ts.patterns`` module.


.. _setup:

Setup
-----

.. note::

    -   M = number of time series
    -   N = length of time series
    -   R = rate of missing values chosen by the user (%); default = 0.2
    -   S = offset in the beginning of the series (%); default = 0.1

    -   W = R * N


.. raw:: html

   <br><br>


.. _scenario_mono_block:

MONO-BLOCK
----------

**Missing Percentage**

.. note::

    -   ``M ∈ [1%, max]; R ∈ [1%, max]``
    -   The size of a single missing block varies between 1% and (100 - ``S``)% of ``N``.
    -   The starting position is the same and begins at ``S`` and progresses until ``W`` is reached, affecting the first series from the top up to ``M%`` of the dataset.


.. raw:: html

   <br><br>


**Disjoint**

.. note::

    -   ``M ∈ [1, max]; R ∈ [1%, max]``
    -   The size of a single missing block varies between 1% and (100 - ``S``)% of ``N``.
    -   The starting position of the first missing block begins at ``S``.
    -   Each subsequent missing block starts immediately after the previous one ends, continuing this pattern until the limit of the dataset or ``N`` is reached.


.. raw:: html

   <br><br>


**Overlap**

.. note::

    -   ``M ∈ [1, max]; R ∈ [1%, max]``
    -   The size of a single missing block varies between 1% and (100 - ``S``)% of ``N``.
    -   The starting position of the first missing block begins at ``S``.
    -   Each subsequent missing block starts after the previous one ends, but with a shift back of ``X%``, creating an overlap.
    -   This pattern continues until the limit or ``N`` is reached.


.. raw:: html

   <br><br>


**Percentage Shift**

.. note::

    -   ``M ∈ [1%, max]; R ∈ [1%, max]``
    -   The size of a single missing block varies between 1% and (100 - ``S``)% of ``N``.
    -   The starting position is randomly shifted by adding a random value to ``S``, then progresses until ``W `` is reached, affecting the first series from the top up to ``M%`` of the dataset.


.. raw:: html

   <br><br>


.. _scenario_multi_block:

MULTI-BLOCK
-----------

**Missing Completely At Random**

.. note::

    -   ``M ∈ [1%, max]; R ∈ [1%, max]``
    -   Data blocks of the same size are removed from arbitrary series at a random position between ``S`` and ``N``, until a total of ``W`` per series is missing.


.. raw:: html

   <br><br>


**Block Distribution**

.. note::

    -   ``M ∈ [1%, max]; R ∈ [1%, max]``
    -   Data is removed following a distribution given by the user for every values of ``N``, affecting the first series from the top up to ``M%`` of the dataset.

