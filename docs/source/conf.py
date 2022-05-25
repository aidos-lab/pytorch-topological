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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'torch_topological'
copyright = '2022, Bastian Rieck'
author = 'Bastian Rieck'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
]

# Ensure that member functoins are documented. These are sane defaults.
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Ensures that modules are sorted correctly. Since they all pertain to
# the same package, the prefix itself can be ignored.
modindex_common_prefix = ['torch_topological.']

# Specifies how to actually find the sources of the modules. Ensures
# that people can jump to files in the repository directly.
def linkcode_resolve(domain, info):
    # Let's frown on global imports and do everything locally as much as
    # we can.
    import sys
    import torch_topological

    if domain != 'py':
        return None
    if not info['module']:
        return None

    # Attempt to identify the source file belonging to an `info` object.
    # This code is adapted from the Sphinx configuration of `numpy`; see
    # https://github.com/numpy/numpy/blob/main/doc/source/conf.py.
    def find_source_file(module):
        obj = sys.modules[module]

        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)

        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(
            fn,
            start=os.path.dirname(torch_topological.__file__)
        )

        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    try:
        module = info['module']
        source = find_source_file(module)
    except Exception:
        source = None

    root = f'https://github.com/aidos-lab/pytorch-topological/tree/main/{project}/'

    if source is not None:
        fn, start, end = source
        return root + f'{fn}#L{start}-L{end}'
    else:
        return None
