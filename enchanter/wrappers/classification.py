"""
.. warning:: `wrappers.classification` package has been renamed to `tasks` since v0.7.0 and will be remove in v0.8.0
"""

import warnings

warnings.warn("`wrappers.classification` package has been renamed to `tasks` since v0.7.0"
              "The deprecated package name will be remove in v0.8.0", DeprecationWarning)

from enchanter.tasks.classification import ClassificationRunner     # noqa: F403

