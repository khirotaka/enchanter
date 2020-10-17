:tocdepth: 3

enchanter.tasks
==========================


.. automodule:: enchanter.tasks
.. currentmodule:: enchanter.tasks


Classification Task
--------------------

`ClassificationRunner`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ClassificationRunner
    :members:
    :show-inheritance:

    .. rubric:: Methods
    .. autosummary::
        ~general_step
        ~general_end


Regression Task
---------------

`RegressionnRunner`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RegressionRunner
    :members:
    :show-inheritance:

    .. rubric:: Methods
    .. autosummary::
        ~general_step
        ~general_end

`AutoEncoderRunner`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AutoEncoderRunner
    :members:
    :show-inheritance:

    .. rubric:: Methods
    .. autosummary::
        ~general_step
        ~general_end


Time Series Representation Learning Task
---------------------------------------------

`TimeSeriesUnsupervisedRunner`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TimeSeriesUnsupervisedRunner
    :members:
    :show-inheritance:

    .. rubric:: Methods
    .. autosummary::
        ~calculate_negative_loss_per_negative_sample
        ~calculate_negative_loss
        ~encode
        ~train_step
        ~train_end


Ensemble
----------------

`BaseEnsembleEstimator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BaseEnsembleEstimator
    :members:
    :show-inheritance:

`SoftEnsemble`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SoftEnsemble
    :members:
    :show-inheritance:


`HardEnsemble`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: HardEnsemble
    :members:
    :show-inheritance: