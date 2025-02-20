# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

project = 'imputegap'
copyright = '2025, Quentin Nater'
author = 'Quentin Nater'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
]

autosummary_generate = True  # Automatically generate method summaries
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",  # Previous/Next buttons
        "searchbox.html",
    ]
}


html_logo = "https://www.naterscreations.com/imputegap/logo_imputegab.png"
html_favicon = "https://www.naterscreations.com/imputegap/favicon.png"

html_static_path = ['static']
html_css_files = ['custom.css']

# Set the version and release info
version = '1.0.4'
release = '1.0.4'


# You can also add links to edit the documentation on GitHub
html_theme_options = {
    "description": "A general-purpose imputation library",
    "fixed_sidebar": True,
    "sidebar_width": "220px",
    "page_width": "960px",
    "font_family": "Arial, sans-serif",
    "head_font_family": "Arial, sans-serif",
    "show_relbars": True,
    "body_max_width": "800px",
}




templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = False


html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
}

sys.path.insert(0, os.path.abspath('../../../'))  # Adjust path to the project root
