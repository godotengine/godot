# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'GUT'
copyright = '2024, Butch Wesley'
author = 'bitwes'

release = '9.4.0'
version = '9.4.0 for Godot-4'

# -- General configuration


extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_rtd_dark_mode',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

source_suffix = [
    ".md",
    ".rst",
]

html_static_path = ['_static']

# custom.css copied from Godot's CSS:
# https://github.com/godotengine/godot-docs/blob/4.2/_static/css/custom.css
html_css_files = [
    'css/gut_custom.css'
]

html_js_files = [
]

html_favicon = 'favicon.ico'
html_logo = '_static/images/GutDocsLogo.png'
