========
Patterns
========

.. _setup:

Setup
-----

.. note::

    -   M : number of time series
    -   N : length of time series
    -   R : rate of missing values chosen by the user (%); default = 0.2
    -   W : offset window in the beginning of the series (%); default = 0.1

.. raw:: html

   <br>


.. _scenario_mono_block:

MONO-BLOCK
----------
One missing block per series


**Aligned**

.. note::

    -   The missing blocks are aligned.
    -   ``S ∈ [1%, M]; R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position is the same and begins at ``W`` and progresses until the size of the missing block is reached, affecting the first series from the top up to ``S%`` of the dataset.


.. raw:: html

   <br>


**Disjoint**

.. note::

    -   The missing blocks are disjoint.
    -   ``S ∈ [1, M]; R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position of the first missing block begins at ``W``.
    -   Each subsequent missing block starts immediately after the previous one ends, continuing this pattern until the limit of the dataset or ``N`` is reached.


.. raw:: html

   <br>


**Overlap**:

.. note::

    -   The missing blocks are overlapping.
    -   ``S ∈ [1, M]; R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position of the first missing block begins at ``W``.
    -   Each subsequent missing block starts after the previous one ends, but with a shift back of ``X%``, creating an overlap.
    -   This pattern continues until the limit or ``N`` is reached.


.. raw:: html

   <br>


**Scattered**

.. note::

    -   The missing blocks are scattered.
    -   ``S ∈ [1%, M]; R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position is randomly shifted by adding a random value to ``W``, then progresses until the size of the missing block is reached, affecting the first series from the top up to ``S%`` of the dataset.


.. raw:: html

   <br>


.. _scenario_multi_block:

MULTI-BLOCK
-----------

Multiple missing blocks per series


**MCAR**

.. note::

    -   The blocks are missing completely at random
    -   ``S ∈ [1%, M]; R ∈ [1%, (100-W)%]``
    -   Data blocks of the same size are removed from arbitrary series at a random position between ``W`` and ``N``, until the total number of missing values per series is reached.


.. raw:: html

   <br>


**Block Distribution**

.. note::

    -   The missing blocks follow a distribution.
    -   ``S ∈ [1%, M]; R ∈ [1%, (100-W)%]``
    -   Data is removed following a distribution given by the user for every values of the series, affecting the first series from the top up to ``S%`` of the dataset.

If you need to remove data following a specific distribution, please refer to this `tutorial <tutorials_distribution.html>`_.

