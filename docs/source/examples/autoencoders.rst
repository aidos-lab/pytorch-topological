Autoencoders with ``torch_topological``
=======================================

In this example, we will create a simple autoencoder based on the
*Topological Signature Loss* introduced by Moor et al. [Moor20a]_.

A simple autoencoder
--------------------

We first define a simple linear autoencoder. The representations
obtained from this autoencoder are very similar to those obtained via
PCA.

.. literalinclude:: ../../../torch_topological/examples/autoencoders.py
   :language: python
   :pyobject: LinearAutoencoder

Of particular interest in the code are the `encode` and `decode`
functions. With ``encode``, we *embed* data in a latent space, whereas
with ``decode``, we reconstruct it to its 'original' space.

This reconstruction is of course never perfect. We therefore measure is
quality using a reconstruction loss. Let's zoom into the specific
function for this:

.. literalinclude:: ../../../torch_topological/examples/autoencoders.py
   :language: python
   :pyobject: LinearAutoencoder.forward

The important take-away here is that ``forward`` should return at least
return one *loss value*. We will make use of this later on!

A topological wrapper for autoencoder models
--------------------------------------------

Our previous model uses ``encode`` to provide us with a lower-dimensional
representation, the so-called *latent representation*. We can use this
representation in order to calculate a topology-based loss! To this end,
let's write a new ``forward`` function that uses an existing model ``model``
for the latent space generation:

.. literalinclude:: ../../../torch_topological/examples/autoencoders.py
   :language: python
   :pyobject: TopologicalAutoencoder.forward

In the code above, the important things are:

1. The use of a Vietoris--Rips complex ``self.vr`` to obtain persistence
   information about the input space ``x`` and the latent space ``z``,
   respectively. We call this type of data ``pi_x`` and ``pi_z``,
   respectively.

2. The call to a topology-based loss function ``self.loss()``, which takes
   two spaces ``x`` and ``y``, as well as their corresponding persistence
   information, to calculate the *signature loss* from [Moor20a]_.

Putting this all together, we have the following 'wrapper class' that
makes an existing model topology-aware:

.. literalinclude:: ../../../torch_topological/examples/autoencoders.py
   :language: python
   :pyobject: TopologicalAutoencoder

See [Moor20a]_ for more models to extend---being topology-aware can be
crucial for many applications. 

Source code
-----------

Here's the full source code of this example.

.. literalinclude:: ../../../torch_topological/examples/autoencoders.py
   :language: python
