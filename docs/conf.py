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
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'ndindex'
copyright = '2020, Quansight'
author = 'Quansight'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
]

import commonmark

# From
# https://stackoverflow.com/questions/56062402/force-sphinx-to-interpret-markdown-in-python-docstrings-instead-of-restructuredt

def docstring(app, what, name, obj, options, lines):
    md  = '\n'.join(lines)
    ast = commonmark.Parser().parse(md)
    rst = commonmark.ReStructuredTextRenderer().render(ast)
    lines.clear()
    lines += rst.splitlines()

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# Make warnings errors

# Make sphinx give errors for bad cross-references
nitpicky = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

html_theme_options = {
    'github_user': 'Quansight',
    'github_repo': 'ndindex',
    'github_banner': True,
    # 'logo': 'logo.jpg',
    # 'logo_name': True,
    # 'show_related': True,
    # Needs a release with https://github.com/bitprophet/alabaster/pull/101 first
    'show_relbars': True,

    # Colors

    'base_bg': '#EEEEEE',
    'narrow_sidebar_bg': '#DDDDDD',
    # Sidebar text
    'gray_1': '#000000',
    'narrow_sidebar_link': '#333333',
    # Doctest background
    'gray_2': '#F0F8FF',

    # Remove gray background from inline code
    'code_bg': '#EEEEEE',

    # Originally 940px
    'page_width': '1000px',

    # Fonts
    'font_family': "Palatino, 'goudy old style', 'minion pro', 'bell mt', Georgia, 'Hiragino Mincho Pro', serif",
    'font_size': '18px',
    'code_font_family': "'Menlo', 'Deja Vu Sans Mono', 'Consolas', 'Bitstream Vera Sans Mono', monospace",
    'code_font_size': '0.85em',
    }

html_sidebars = {
    '**': ['globaltocindex.html', 'searchbox.html'],
}

mathjax_config = {
  'TeX': {
      'equationNumbers': {
          'autoNumber': "AMS",
      },
  },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Lets us use single backticks for code
default_role = 'code'
