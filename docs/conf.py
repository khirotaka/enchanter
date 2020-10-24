# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_material
sys.path.insert(0, os.path.abspath('../'))

import enchanter

# -- Project information -----------------------------------------------------

project = 'Enchanter'
copyright = '2020, Hirotaka Kawashima'
author = 'Hirotaka Kawashima'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_material"
]
html_show_sourcelink = False


autoapi_dirs = ['../enchanter']
autoapi_generate_api_docs = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to reference directory, that match files and
# directories to ignore when looking for reference files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_title = "Enchanter documentation"
html_theme = "sphinx_material"
html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()
html_logo = "_static/images/Enchanter-Logo-clear.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autoclass_content = 'both'
master_doc = 'index'

highlight_language = "python"

html_theme_options = {
    'nav_title': 'Enchanter',
    'color_primary': 'indigo',
    'color_accent': 'light-blue',
    'repo_url': 'https://github.com/khirotaka/enchanter',
    'repo_name': 'Enchanter',
    'heroes': {
        "index": "Enchanter is a library for machine learning tasks for comet.ml users."
    },
    "theme_color": "#2196f3",
    'html_minify': True,
    'css_minify': True,

    "version_dropdown": True,
    "version_json": "_static/versions.json",
    "version_info": {
        "master": "https://enchanter.readthedocs.io/en/latest/",
        "develop": "https://enchanter.readthedocs.io/en/develop/",
        "v0.5.0": "https://enchanter.readthedocs.io/en/v0.5.0/",
        "v0.5.1": "https://enchanter.readthedocs.io/en/v0.5.1/",
        "v0.5.2": "https://enchanter.readthedocs.io/en/v0.5.2/",
        "v0.5.3": "https://enchanter.readthedocs.io/en/v0.5.3/",
        "v0.6.0": "https://enchanter.readthedocs.io/en/v0.6.0/",
        "v0.7.0": "https://enchanter.readthedocs.io/en/v0.7.0/",
        "v0.7.1": "https://enchanter.readthedocs.io/en/v0.7.1/",
        "v0.8.0": "https://enchanter.readthedocs.io/en/v0.8.0/",
        "v0.8.1": "https://enchanter.readthedocs.io/en/v0.8.1/",
    },
}

html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html", "globaltoc.html"]
}
