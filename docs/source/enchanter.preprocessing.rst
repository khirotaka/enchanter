:tocdepth: 4

enchanter.preprocessing
==============================


Signal
-------------------------------

Transforms
~~~~~~~~~~~~~~~~~~~~

.. automodule:: enchanter.preprocessing.signal.transforms
.. currentmodule:: enchanter.preprocessing.signal.transforms

transforms.Compose
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Compose
    :members:
    :show-inheritance:

    .. rubric:: Methods
    .. autosummary::
        ~insert
        ~append
        ~extend


transforms.FixedWindow
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FixedWindow
    :members:
    :show-inheritance:


transforms.GaussianNoise
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GaussianNoise
    :members:
    :show-inheritance:

transforms.RandomScaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RandomScaling
    :members:
    :show-inheritance:


transforms.Pad
^^^^^^^^^^^^^^^^^^

.. autoclass:: Pad
    :members:
    :show-inheritance:



Functions
~~~~~~~~~~~~~~~~~~~~


.. automodule:: enchanter.preprocessing.signal
.. currentmodule:: enchanter.preprocessing.signal

`FixedSlidingWindow`
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FixedSlidingWindow
    :members:
    :show-inheritance:

    .. rubric:: Methods
    .. autosummary::
        ~transform
        ~clean


`adjust_sequences`
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: adjust_sequences
