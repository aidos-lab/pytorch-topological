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

Source code
-----------

Here's the full source code of this example.

.. literalinclude:: ../../../torch_topological/examples/summary_statistics.py
   :language: python
