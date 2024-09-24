# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Memristor Crossbar Training class'
copyright = '2024, Caterina Baldassini'
author = 'Caterina Baldassini'
release = '27/07/2024'

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'myst_parser'
    ]

templates_path = ['_templates']
exclude_patterns = []

myst_enable_extensions = ["amsmath", "dollarmath"]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']



# HTML options
# html_theme_options = {
#     'collapse_navigation': False,
#     'sticky_navigation': True,
#     'navigation_depth': 4,
#     'includehidden': True,
#     'titles_only': False
# }

# Other options
html_title = 'Memristor Crossbar Training class Documentation'
html_short_title = 'Memristor Crossbar Docs'
html_logo = '_static/your_logo1.png'
html_favicon = '_static/favicon.ico'
html_show_sphinx = False