project = "MACE-SCF"
author = "ACEsuit"
copyright = "2026, ACEsuit"

extensions = [
    "myst_parser",
]

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "developer/**",
    "examples/**",
    "reference/**",
]

html_theme = "sphinx_rtd_theme"
html_title = "MACE-SCF"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
