# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "LimboAI"
copyright = "2021-present Serhii Snitsaruk and the LimboAI contributors"
author = "Serhii Snitsaruk and the LimboAI contributors"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_rtd_dark_mode", "sphinx_copybutton"]

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- Markdown configuration (sphinx_markdown_builder).
# markdown_anchor_sections = True
# markdown_anchor_signatures = True
# markdown_docinfo = ""
# markdown_http_base = "https://your-domain.com/docs"
# markdown_uri_doc_suffix = ".html"

# -- Dark-mode configuration.
default_dark_mode = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "collapse_navigation": True,
}
