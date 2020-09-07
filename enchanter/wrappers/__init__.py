"""
.. warning:: `wrappers` package has been renamed to `tasks` since v0.7.0 and will be remove in v0.8.0
"""

import warnings


warnings.warn(
    "`wrappers` package has been renamed to `tasks` since v0.7.0"
    "The deprecated package name will be remove in v0.8.0",
    DeprecationWarning,
)

from enchanter.tasks import *  # noqa: F403
