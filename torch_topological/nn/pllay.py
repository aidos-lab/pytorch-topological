"""PLLay module.

The following is an implementation of the persistence landscape-based
topological layer (PLLay) described in Definition 3.1 of [1].

References
----------
.. [1] K. Kim et al., "PLLay: Efficient Topological Layer
based on Persistence Landscapes", *34th Conference on Neural Information
Processing Systems (NeurIPS 2020)*, Vancouver, Canada.
"""

import math
import torch
from torch.nn import Parameter, Linear, init
from torch.nn.functional import relu, softmax
from torch.nn.utils.rnn import pad_sequence
from torch_topological.nn import PersistenceInformation


class PLLay(torch.nn.Module):
    """A neural network layer PLLay.

    This layer consists of q structure elements which are concatenated.
    Each structure element processes persistence information by using
    a learned weighted sum of persistence landscapes, evaluating them on
    points, then mapping these values to a real number using a learnable
    function.  See [1] for details.

    References
    ----------
    .. [1] K. Kim et al., "PLLay: Efficient Topological Layer based on
    Persistence Landscapes", *34th Conference on Neural Information
    Processing Systems (NeurIPS 2020)*, Vancouver, Canada.
    """

    def __init__(self, output_dim, K_max, t_min, t_max, m, layer="Affine"):
        """Initialize new module.

        Parameters
        ----------
        output_dim : int
            The dimension of the features of the output, which is the
            number of structure elements.

        K_max : int
            The number of persistence landscapes the model considers.
            The model only considers K_max top landscapes.  If K_max is
            larger than the number of barcodes, error may arise.

        t_min : float
            The lower bound of the points evaluated on the weighted
            landscapes.

        t_max : float
            The upper bound of the points evaluated on the weighted
            landscapes.

        m : int
            The number of points evaluated on the weighted landscapes.
            m evenly spaced points between [t_min, t_max] is evaluated
            on the weighted landscapes.

        layer : str or Callable
            The final layer mapping the m dimensional features to
            values.  If layer is a string, it has to be "Affine" or
            "Logarithmic", which provides an affine or logarithmic
            transformation as in [1].  Layer can also be a Callable
            which takes in tensors of shape [B, output_dim, m] as input.
            Default is "Affine".

        References
        ----------
        .. [1] K. Kim et al., "PLLay: Efficient Topological Layer based
        on Persistence Landscapes", *34th Conference on Neural
        Information Processing Systems (NeurIPS 2020)*, Vancouver,
        Canada.
        """
        super().__init__()
        self.weight = Parameter(torch.empty(output_dim, K_max))
        if isinstance(layer, str):
            self.layer = layer
            if layer == "Affine":
                self.bias = Parameter(torch.empty(output_dim))
                self.linear = Linear(m, output_dim, bias=False)
            elif layer == "Logarithmic":
                self.mu = Parameter(torch.empty(output_dim, m))
                self.sigma = Parameter(torch.empty(output_dim))
            else:
                raise ValueError("Layer must either be 'Affine', "
                                 "'Logarithmic' or a Callable.")
        else:
            self.layer = layer
        self.t = torch.linspace(t_min, t_max, m)
        self.k = K_max
        self.m = m
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        if self.layer == "Affine":
            bound = 1 / math.sqrt(self.m)
            init.uniform_(self.bias, a=-bound, b=bound)
        if self.layer == "Logarithmic":
            # follows the init in torchph.nn.slayer.SLayerExponential
            init.uniform_(self.mu, a=0., b=1.)
            init.ones_(self.sigma)

    def forward(self, pers_info):
        """Implement forward pass.

        Parameters
        ----------
        x : List[List[PersistenceInformation]] or
        List[PersistenceInformation] or Tensor
            Input persistence information.  If batched, input should be
            list of lists of PersistenceInformation or a tensor of
            diagrams of shape (B, N, 2).  If not batched, input should
            be list of PersistenceInformation or a tensor of diagram of
            shape (N, 2).

        Returns
        -------
        Tensor
            Tensor of shape (B, output_dim) or (output_dim, ) if not
            batched.
        """
        diagram = self.preprocessing(pers_info)
        x = diagram[..., 0][..., None]
        y = diagram[..., 1][..., None]
        triangles = relu(torch.minimum(self.t - x, y - self.t))
        landscapes = torch.topk(triangles, self.k, dim=-2, sorted=True).values
        weighted_landscapes = torch.matmul(softmax(self.weight, dim=-1),
                                           landscapes)
        if self.layer == "Affine":
            output = torch.diagonal(self.linear(weighted_landscapes),
                                    dim1=-2, dim2=-1) + self.bias
            return output
        elif self.layer == "Logarithmic":
            norm = torch.norm(weighted_landscapes - self.mu, dim=-1)
            output = torch.exp(-1 * self.sigma * norm)
            return output
        else:
            output = self.layer(output)
            return output

    def preprocessing(self, x):

        if isinstance(x, list):
            if isinstance(x[0], PersistenceInformation):
                return self.select_diagrams_by_dim(x)
            elif isinstance(x[0], list):
                if isinstance(x[0][0], PersistenceInformation):
                    diagrams = [self.select_diagrams_by_dim(x_) for x_ in x]
                    return pad_sequence(diagrams,
                                        batch_first=True,
                                        padding_value=0.)
                else:
                    raise TypeError("Input must be "
                                    "list of PersistenceInformation "
                                    "or list of lists or tensors.")
            else:
                raise TypeError("Input must be "
                                "list of PersistenceInformation or "
                                "list of lists or tensors.")

        else:
            return x

    def select_diagrams_by_dim(self, x):  # adapted from SelectByDimension

        diagrams = []

        for pers_info in x:
            if (
                (pers_info.dimension == self.dim)
                and (len(pers_info.diagram) != 0)
            ):
                diagrams.append(pers_info.diagram)

        if len(diagrams) > 1:
            return torch.cat(diagrams, dim=0)
        elif len(diagrams) == 1:
            return diagrams[0]
        else:
            raise ValueError("Input does not have diagrams "
                             "in the given dimension.")
