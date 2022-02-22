<img src="torch_topological.svg" height=128 alt="`pytorch-topological` icon" />

# `pytorch-topological`: A topological machine learning framework for `pytorch`

[![Documentation](https://readthedocs.org/projects/pytorch-topological/badge/?version=latest)](https://pytorch-topological.readthedocs.io/en/latest/?badge=latest) ![PyPI - License](https://img.shields.io/pypi/l/torch_topological) ![PyPI](https://img.shields.io/pypi/v/torch_topological)

`pytorch-topological` (or `torch_topological`) is a topological machine
learning framework for [PyTorch](https://pytorch.org). It aims to
collect *loss terms* and *neural network layers* in order to simplify
building the next generation of topology-based machine learning tools.

# Topological machine learning in a nutshell 

*Topological machine learning* refers to a new class of machine learning
algorithms that are able to make use of topological features in data
sets. In contrast to methods based on a purely geometrical point of
view, topological features are capable of focusing on *connectivity
aspects* of a data set. This provides an interesting fresh perspective
that can be used to create powerful hybrid algorithms, capable of
yielding more insights into data.

This is an *emerging research field*, firmly rooted in computational
topology and topological data analysis. If you want to learn more about
how topology and geometry can work in tandem, here are a few resources
to get you started:

- Amézquita et al., [*The Shape of Things to Come: Topological Data Analysis and Biology,
  from Molecules to Organisms*](https://doi.org/10.1002/dvdy.175), Developmental Dynamics
  Volume 249, Issue 7, pp. 816--833, 2020.

- Hensel et al., [*A Survey of Topological Machine Learning Methods*](https://www.frontiersin.org/articles/10.3389/frai.2021.681108/full),
  Frontiers in Artificial Intelligence, 2021.

# Installation

It is recommended to use the excellent [`poetry`](https://python-poetry.org) framework
to install `torch_topological`:

```
poetry add torch-topological
```

Alternatively, use `pip` to install the package:

```
pip install -U torch-topological
```

# Usage

`torch_topological` is still a work in progress. You can [browse the documentation](https://pytorch-topological.readthedocs.io)
or, if code reading is more your thing, dive directly into [some example
code](./torch_topological/examples).

# Contributing

Check out the [contribution guidelines](CONTRIBUTING.md) or the [road
map](ROADMAP.md) of the project.

# Acknowledgements

Our software and research does not exist in a vacuum. `pytorch-topological` is standing
on the shoulders of proverbial giants. In particular, we want to thank the
following projects for constituting the technical backbone of the
project:

| [`giotto-tda`](https://github.com/giotto-ai/giotto-tda)       | [`gudhi`](https://github.com/GUDHI/gudhi-devel)<br />       |
|---------------------------------------------------------------|-------------------------------------------------------------|
| <img src="logos/giotto.jpg" height=128 alt="`giotto` icon" /> | <img src="logos/gudhi.png" height=128 alt="`GUDHI` icon" /> |

Furthermore, `pytorch-topological` draws inspiration from several
projects that provide a glimpse into the wonderful world of topological
machine learning:

- [`difftda`](https://github.com/MathieuCarriere/difftda) by [Mathieu Carrière](https://github.com/MathieuCarriere)

- [`Ripser`](https://github.com/Ripser/ripser) by [Ulrich Bauer](https://github.com/ubauer)

- [`TopologyLayer`](https://github.com/bruel-gabrielsson/TopologyLayer) by [Rickard Brüel Gabrielsson](https://github.com/bruel-gabrielsson)

- [`topological-autoencoders`](https://github.com/BorgwardtLab/topological-autoencoders) by [Michael Moor](https://github.com/mi92), [Max Horn](https://github.com/ExpectationMax), and [Bastian Rieck](https://github.com/Pseudomanifold)

- [`torchph`](https://github.com/c-hofer/torchph) by [Christoph Hofer](https://github.com/c-hofer) and [Roland Kwitt](https://github.com/rkwitt)

Finally, `pytorch-topological` makes heavy use of [`POT`](https://pythonot.github.io), the Python Optimal Transport Library.
We are indebted to the many contributors of all these projects.
