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

# Dependencies

`torch_topological` is making heavy use of [`giotto-ph`](https://github.com/giotto-ai/giotto-ph),
a high-performance implementation of [`Ripser`](https://github.com/Ripser/ripser).

# Acknowledgements

Our software and research does not exist in a vacuum. `pytorch-topological` is standing
on the shoulders of proverbial giants. In particular, we want to the
following projects:

[`giotto-tda`](https://github.com/giotto-ai/giotto-tda)
<img src="logos/giotto.jpg" height=128 alt="`giotto` icon" />
