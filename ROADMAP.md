# What's the future of `pytorch-topological`?

---

**Vision** `pytorch-topological` aims to be the first stop for building
powerful applications using *topological machine learning* algorithms,
i.e. algorithms that are capable of jointly leveraging geometrical and
topological features in a data set

---

To make this vision a reality, we first and foremost need to rely on
exceptional documentation. It is not enough to write outstanding code;
we have to demonstrate the power of topological algorithms to our users
by writing well-documented code and contributing examples.

Here are short-term and long-term goals, roughly categorised:

## API

- [ ] Provide consistent way of handling batches or tensor inputs. Most
  of the modules rely on *sparse* inputs as lists.
- [ ] Support different backends for calculating persistent homology. At
  present, we use [`GUDHI`](https://github.com/GUDHI/gudhi-devel/) for
  cubical complexes and [`giotto-ph`](https://github.com/giotto-ai/giotto-ph)
  for Vietoris--Rips complexes. It would be nice to be able to swap
  implementations easily.
- [ ] Check out the use of sparse tensors; could be a potential way
  forward for representing persistence information. The drawback is that
  we cannot fill everything with zeroes; there has to be a way to
  indicate 'unset' information.

## Distances and kernels

At present, the module supports Wasserstein distance calculations and
bottleneck distance calculations between persistence diagrams. In
addition to this, several 'pseudo-distances' based on summary statistics
have been implemented. There are numerous kernels out there that could
be included:

- [ ] The multi-scale kernel by [Reininghaus et al.](https://openaccess.thecvf.com/content_cvpr_2015/papers/Reininghaus_A_Stable_Multi-Scale_2015_CVPR_paper.pdf)
- [ ] The sliced Wasserstein distance kernel by [Carrière et al.](https://arxiv.org/abs/1706.03358)

This list is **incomplete**.

## Layers

There are quite a few topology-based layers that have been proposed by
members of the community. We should include all of them to make them
available with a single, consistent interface.

- [ ] Include [`PersLay`](https://github.com/MathieuCarriere/perslay).
  This requires a conversion from TensorFlow code.
- [ ] Include [`PLLay`](https://github.com/jisuk1/pllay).
  This requires a conversion from TensorFlow code.
- [ ] Include [`SLayer`](https://github.com/c-hofer/torchph). This is
  still an ongoing effort.
- [x] Include [`TopologyLayer`](https://github.com/bruel-gabrielsson/TopologyLayer).

## Loss terms

- [x] Include *signature loss* from [`topological-autoencoders`](https://github.com/BorgwardtLab/topological-autoencoders)
