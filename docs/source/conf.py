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
    'myst_parser',
    "sphinx.ext.viewcode", 
    "sphinx.ext.linkcode",
    "sphinx_copybutton",
    ]

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/') + '.py'
    start_line = info.get('lineno', 1)
    end_line = start_line + info.get('linespan', 0)
    return f"https://github.com/cateru/Memristor_Crossbar_Training_Class/blob/main/{filename}#L{start_line}-L{end_line}" if end_line > start_line else f"https://github.com/cateru/Memristor_Crossbar_Training_Class/blob/main/{filename}#L{start_line}"

templates_path = ['_templates']
exclude_patterns = []

myst_enable_extensions = ["amsmath", "dollarmath"]

autodoc_mock_imports = ["experimental_conductances", "Memristor_Crossbar"]
pygments_style = "monokai"


source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css',]

copybutton_image_svg = """
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="24"
  height="24"
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
>
  <path d="M14 3v4a1 1 0 0 0 1 1h4" />
  <path d="M17 21h-10a2 2 0 0 1 -2 -2v-14a2 2 0 0 1 2 -2h7l5 5v11a2 2 0 0 1 -2 2z" />
  <path d="M9 17h6" />
  <path d="M9 13h6" />
</svg>
"""


# HTML options
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'logo_only' : True,
    'includehidden': True,
    'titles_only': False,
    "body_max_width": "none",
}


# Other options
html_title = 'Memristor Crossbar Training class Documentation'
html_short_title = 'Memristor Crossbar Docs'
html_logo = '_static/logo_new_2.png'
html_favicon = '_static/favicon.ico'
html_show_sphinx = False