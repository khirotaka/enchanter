__version__ = "0.9.0"
__author__ = "Hirotaka Kawashima"
__author_email__ = "khirotaka@vivaldi.net"
__license__ = "Apache-2.0"
__copyright__ = "Copyright (c) 2020-2021, {}.".format(__author__)
__homepage__ = "https://github.com/khirotaka/enchanter"

__docs__ = "Enchanter is a library for machine learning tasks for comet.ml users."


from enchanter import addons
from enchanter import engine
from enchanter import tasks


__all__ = ["addons", "engine", "tasks"]
