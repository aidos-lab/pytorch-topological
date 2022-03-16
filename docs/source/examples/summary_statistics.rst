Point cloud optimisation with summary statistics
================================================

One interesting use case of ``torch_topological`` involves changing the
shape of a point cloud using topological summary statistics. Such
summary statistics can *either* be used as simple loss functions,
constituting a computationally cheap way of assessing the topological
similarity of a given point cloud to a target point cloud, *or*
serve to highlight certain topological properties of a single point
cloud.

In this example, we will consider *both* operations.

Ingredients
-----------

Our main ingredient is the :py:class:`torch_topological.nn.SummaryStatisticLoss`
class. This class bundles different summary statistics on persistence
diagrams and permits their calculation and comparison.

This class can operate in two modes:

1. Calculating the loss for a single input data set.
2. Calculating the loss difference for two input data sets.

Our example will showcase both of these modes!

Optimising all the point clouds
-------------------------------

Here's the bulk of the code required to optimise a point cloud. We will
walk through the most important parts!

.. literalinclude:: ../../../torch_topological/examples/summary_statistics.py
   :language: python
   :pyobject: main 

Next to creating some test data sets---check out :py:mod:`torch_topological.data`
for more routines---the most important thing is to make sure that ``X``,
our point cloud, is a trainable parameter.

With that being out of the way, we can set up the summary statistic loss
and start training. The main loop of the training might be familiar to
those of you that already have some experience with ``pytorch``: it
merely evaluates the loss and optimises it, following a general
structure:

.. code-block::

  # Set up your favourite optimiser
  opt = optim.SGD(...)

  for i in range(100):

    # Do some calculations and obtain a loss term. In our specific
    # example, we have to get persistence information from data and
    # evaluate the loss based on that. 
    loss = ...

    # This is what you will see in many such examples: we set all
    # gradients to zero and do a backwards pass.
    opt.zero_grad()
    loss.backward()
    opt.step()

The rest of this example just involves some nice plotting.

Source code
-----------

Here's the full source code of this example.

.. literalinclude:: ../../../torch_topological/examples/summary_statistics.py
   :language: python
