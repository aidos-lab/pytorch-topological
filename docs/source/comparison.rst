Comparison with other packages
==============================

If you are already familiar with certain packages for calculating
topological features, you might be interested in understanding in
what aspects ``torch_topological`` differs from them. This is not
meant to be a comprehensive comparison; we are aiming for a brief
overview to simplify getting acquainted with the project.

``giotto-tda``
--------------

`giotto-tda <https://giotto-ai.github.io/gtda-docs>`_ is a flagship
package, developed by numerous members of L2F. Its primary goal is to
provide an interface consistent with ``scikit-learn``, thus facilitating
an integration of topological features into a data science workflow.

By contrast, ``torch_topological`` is meant to simplify the development
of hybrid algorithms that can be easily integrated into deep learning 
architectures. ``giotto-tda`` is developed by a large team with a much
more professional development agenda, whereas ``torch_topological`` is
geared more towards researchers that want to prototype the integration
of topological features.

``TopologyLayer``
-----------------

`TopologyLayer <https://github.com/bruel-gabrielsson/TopologyLayer>`_ is
a library developed by Rickard Brüel Gabrielsson and others,
accompanying their AISTATS publication `A Topology Layer for Machine Learning <https://proceedings.mlr.press/v108/gabrielsson20a.html>`_.

``torch_topological`` subsumes the functionality of ``TopologyLayer``,
albeit under different names:

- :py:class:`torch_topological.nn.VietorisRipsComplex` or
  :py:class:`torch_topological.nn.CubicalComplex` can be used to extract
  topological features from point clouds and images, respectively.

- The ``BarcodePolyFeature`` and ``SumBarcodeLengths`` classes are
  incorporated as summary statistics loss functions instead. See the
  following example for more details: :doc:`examples/summary_statistics`

- The ``PartialSumBarcodeLengths`` function is not implemented, mostly
  because a similar effect can be achieved by pruning the persistence
  diagram manually. This functionality might be added later on.
