:tocdepth: 2

enchanter.engine
========================


.. automodule:: enchanter.engine
.. currentmodule:: enchanter.engine

BaseRunner
----------------------------

.. autoclass:: BaseRunner
    :members:
    :show-inheritance:

    .. rubric:: Methods
    .. autosummary::
        ~add_loader
        ~backward
        ~fit
        ~freeze
        ~initialize
        ~log_hyperparams
        ~predict
        ~quite
        ~run
        ~train_step
        ~train_end
        ~train_cycle
        ~train_config
        ~test_step
        ~test_end
        ~test_cycle
        ~unfreeze
        ~update_optimizer
        ~update_scheduler
        ~val_step
        ~val_end
        ~val_cycle


RunnerIO
----------------------------

.. autoclass:: RunnerIO
    :members:
    :show-inheritance:

    .. rubric:: Methods
    .. autosummary::
        ~model_name
        ~save_checkpoint
        ~load_checkpoint
        ~save
        ~load


enchanter.engine.modules
==============================

.. automodule:: enchanter.engine.modules
.. currentmodule:: enchanter.engine.modules


is_jupyter
----------------------------

.. autofunction:: is_jupyter


get_dataset
----------------------------

.. autofunction:: get_dataset

fix_seed
----------------------------

.. autofunction:: fix_seed

send
----------------------------

.. autofunction:: send

is_tfds
----------------------------

.. autofunction:: is_tfds

tfds_to_numpy
----------------------------

.. autofunction:: tfds_to_numpy
