:tocdepth: 3

enchanter.callbacks
===========================

.. automodule:: enchanter.callbacks
.. currentmodule:: enchanter.callbacks


A set of functions to be executed during training, verification and testing.

Callback
-------------

`Callback`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Callback
    :members:
    :undoc-members:
    :show-inheritance:


CallbackManager
------------------

`CallbackManager`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CallbackManager


EarlyStopping
-------------

`EarlyStopping`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EarlyStopping
    :members:
    :show-inheritance:

    .. rubric:: Methods
    .. autosummary::
        ~check_metrics
        ~on_epoch_end


`EarlyStoppingForTSUS`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EarlyStoppingForTSUS
    :members:
    :show-inheritance:

    .. rubric:: Methods
    .. autosummary::
        ~cross_val
        ~encode
        ~on_epoch_end


------------

Logging
-------------
Provides an alternative logging method to ``comet_ml.Experiment``.



`BaseLogger`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BaseLogger
    :members:
    :show-inheritance:



`TensorBoardLogger`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TensorBoardLogger
    :members:
    :undoc-members:
    :show-inheritance:
