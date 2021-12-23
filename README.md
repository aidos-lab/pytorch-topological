<img src="torch_topological.svg" height=128 alt="`pytorch-topological` icon" />

# `pytorch-topological`: A topological machine learning framework for `pytorch`

![PyPI - License](https://img.shields.io/pypi/l/torch_topological) ![PyPI](https://img.shields.io/pypi/v/torch_topological)

`pytorch-topological` (or `torch_topological`) is a topological machine
learning framework for [PyTorch](https://pytorch.org). It aims to
collect *loss terms* and *neural network layers* in order to simplify
building the next generation of topology-based machine learning tools.

`torch_topological` is still a work in progress. Stay tuned for more
information.

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
