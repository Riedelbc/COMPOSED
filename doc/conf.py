#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# composed documentation build configuration file, created by
# sphinx-quickstart on Mon May  8 23:14:35 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
#
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'composed'
copyright = '2017, Brandalyn Lyjak'
author = 'Brandalyn Lyjak'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '0.1'
# The full version, including alpha/beta/rc tags.
release = version

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
     # The paper size ('letterpaper' or 'a4paper').
     #
     # 'papersize': 'letterpaper',

     # The font size ('10pt', '11pt' or '12pt').
     #
     # 'pointsize': '10pt',

     # Additional stuff for the LaTeX preamble.
     #
     # 'preamble': '',

     # Latex figure (float) alignment
     #
     # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'composed.tex', 'composed Documentation',
     'Brandalyn Lyjak', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#
# latex_use_parts = False

# If true, show page references after internal links.
#
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
#
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
#
# latex_appendices = []

# It false, will not define \strong, \code, 	itleref, \crossref ... but only
# \sphinxstrong, ..., \sphinxtitleref, ... To help avoid clash with user added
# packages.
#
# latex_keep_old_macro_names = True

# If false, no module index is generated.
#
# latex_domain_indices = True

import os, sys, shutil, subprocess

api_modules = [('composed', 'composed', ())]

def run_apidoc(app):
    try:
        srcdir = app.builder.srcdir
        cmd_path = 'sphinx-apidoc'
        if hasattr(sys, 'real_prefix'):
            # Check to see if we are in a virtualenv
            # If we are, assemble the path manually
            cmd_path = os.path.abspath(os.path.join(sys.prefix, 'bin',
                                                    'sphinx-apidoc'))

        for m_name, m_path, m_ignore in api_modules:
            moduledir = os.path.abspath(os.path.join(srcdir, "..", m_path))
            assert os.path.exists(moduledir), \
                "path does not exist, %s" % moduledir
            outdir = os.path.join(srcdir, "%s-autodocs" % m_name)
            ignore_paths = [os.path.join(moduledir, m_sub) for m_sub in m_ignore]
            cmd_call = [cmd_path, '-e', '-o', outdir,
                        '--force', moduledir] + ignore_paths
            subprocess.check_call(cmd_call)
    except Exception as e:
        app.warn("Sphinx API documentation failed to build, the following"
                 " exception was raised: %s %s" % (type(e), e))
        remove_apidocs(app, e)

def remove_apidocs(app, exception):
    for m_name, m_path, m_ignore in api_modules:
        outdir = os.path.join(app.builder.srcdir, "%s-autodocs" % m_name)
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

def setup(app):
    app.connect('builder-inited', run_apidoc)
    app.connect('build-finished', remove_apidocs)
