# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'imputegap'
copyright = '2024, Quentin Nater'
author = 'Quentin Nater'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # for Google or NumPy-style docstrings
    'sphinx.ext.viewcode',  # adds links to source code
    'sphinx.ext.autosummary',  # generates method summaries
    'sphinx_rtd_theme',
]

autosummary_generate = True  # Automatically generate summaries

html_logo = "https://www.naterscreations.com/imputegap/logo_imputegab.png"
html_favicon = "https://www.naterscreations.com/imputegap/favicon.png"

html_static_path = ['static']
html_css_files = ['custom.css']

# Set the version and release info
version = '0.1.8'
release = '0.1.8'


# You can also add links to edit the documentation on GitHub
html_context = {
    'display_github': True,  # Integrate GitHub
    'github_user': 'qnater',
    'github_repo': 'https://github.com/eXascaleInfolab/ImputeGAP',
    'github_version': 'https://github.com/eXascaleInfolab/ImputeGAP',
}


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['static']


import os
import sys
sys.path.insert(0, os.path.abspath('../../../'))  # Adjust path to the project root
