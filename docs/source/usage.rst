Installing and using ``torch_topological``
==========================================

Requirements
------------

``torch_topological`` requires at least Python 3.9. Normally, version
resolval should work automatically. The precise mechanism for this
depends on your installation method (see below).

Installation via ``pip``
------------------------

We recommended installing ``torch_topological`` using ``pip``. This way,
you will always get a release version with a known set of features. It
is *recommended* to use a virtual environment manager such as `poetry <https://python-poetry.org/>`_
for handling the dependencies of your project.

.. code-block:: console

   $ pip install torch_topological

Installation from source
------------------------

Installing the package from source requires a virtual environment
manager capable of parsing ``pyproject.toml`` files. With `poetry <https://python-poetry.org/>`_,
for instance, the following steps should be sufficient:

.. code-block:: console

   $ git clone git@github.com:aidos-lab/pytorch-topological.git 
   $ cd pytorch-topological
   $ poetry install
