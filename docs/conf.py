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
copyright = '2020, Quansight Labs'
author = 'Quansight Labs'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
]

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# # From
# # https://stackoverflow.com/questions/56062402/force-sphinx-to-interpret-markdown-in-python-docstrings-instead-of-restructuredt
#
# def docstring(app, what, name, obj, options, lines):
#     import commonmark
#     md  = '\n'.join(lines)
#     ast = commonmark.Parser().parse(md)
#     rst = commonmark.ReStructuredTextRenderer().render(ast)
#     lines.clear()
#     lines += rst.splitlines()
#
# def setup(app):
#     app.connect('autodoc-process-docstring', docstring)

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# Make warnings errors

# Make sphinx give errors for bad cross-references
nitpicky = True

suppress_warnings = ['toc.circular']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These are defined in _static/custom.css
light_blue = "var(--color-brand-light-blue)"
green = "var(--color-brand-green)"
medium_blue = "var(--color-brand-medium-blue)"
dark_blue = "var(--color-brand-dark-blue)"
dark_bg = "var(--color-brand-dark-bg)"
white = "white"
black = "black"
gray = "#EEEEEE"

theme_colors_common = {
    "color-sidebar-background-border": "var(--color-background-primary)",
    "color-sidebar-brand-text": "var(--color-sidebar-link-text--top-level)",

    "color-admonition-title-background--seealso": "#CCCCCC",
    "color-admonition-title--seealso": black,
    "color-admonition-title-background--note": "#CCCCCC",
    "color-admonition-title--note": black,
    "color-admonition-title-background--warning": "var(--color-problematic)",
    "color-admonition-title--warning": white,
    "admonition-font-size": "var(--font-size--normal)",
    "admonition-title-font-size": "var(--font-size--normal)",

    "color-link-underline--hover": "var(--color-link)",

    "color-api-keyword": "#000000bd",
    "color-api-name": "var(--color-brand-content)",
    "color-api-pre-name": "var(--color-brand-content)",
    "api-font-size": "var(--font-size--normal)",


    }
html_theme_options = {
    'light_logo': 'ndindex_logo_white_bg.svg',
    'dark_logo': 'ndindex_logo_dark_bg.svg',
    "light_css_variables": {
        **theme_colors_common,
        "color-brand-primary": dark_blue,
        "color-brand-content": dark_blue,

        "color-sidebar-background": gray,
        "color-sidebar-item-background--hover": light_blue,
        "color-sidebar-item-expander-background--hover": light_blue,

    },
    "dark_css_variables": {
        **theme_colors_common,
        "color-brand-primary": light_blue,
        "color-brand-content": light_blue,

        "color-api-keyword": "#FFFFFFbd",
        "color-api-overall": "#FFFFFF90",
        "color-api-paren": "#FFFFFF90",

        "color-background-primary": black,

        "color-sidebar-background": dark_bg,
        "color-sidebar-item-background--hover": medium_blue,
        "color-sidebar-item-expander-background--hover": medium_blue,

        "color-highlight-on-target": dark_blue,

        "color-admonition-title-background--seealso": "#555555",
        "color-admonition-title-background--note": "#555555",
        "color-problematic": "#B30000",
    },
    # See https://pradyunsg.me/furo/customisation/footer/
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Quansight-Labs/ndindex",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# custom.css contains changes that aren't possible with the above because they
# aren't specified in the Furo theme as CSS variables
html_css_files = ['custom.css']

sys.path.append(os.path.abspath("./_pygments"))
pygments_style = 'styles.SphinxHighContrastStyle'
pygments_dark_style = 'styles.NativeHighContrastStyle'

html_favicon = "logo/favicon.ico"

myst_enable_extensions = ["dollarmath", "linkify"]

mathjax3_config = {
  'TeX': {
      'equationNumbers': {
          'autoNumber': "AMS",
      },
  },
}

myst_update_mathjax=False

# Lets us use single backticks for code
default_role = 'code'

# Add a header for PR preview builds. See the Circle CI configuration.
if os.environ.get("CIRCLECI") == "true":
    PR_NUMBER = os.environ.get('CIRCLE_PR_NUMBER')
    SHA1 = os.environ.get('CIRCLE_SHA1')
    html_theme_options['announcement'] = f"""This is a preview build from
ndindex pull request <a href="https://github.com/Quansight-Labs/ndindex/pull/{PR_NUMBER}">
#{PR_NUMBER}</a>. It was built against <a
href="https://github.com/Quansight-Labs/ndindex/pull/{PR_NUMBER}/commits/{SHA1}">{SHA1[:7]}</a>.
If you aren't looking for a PR preview, go to <a
href="https://quansight-labs.github.io/ndindex//">the main ndindex documentation</a>. """
