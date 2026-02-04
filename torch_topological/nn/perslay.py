"""PersLay module.

The following is converted from the code given in the official
repository of [1].

References
----------
.. [1] M. Carriere et al., "PersLay: A Neural Network Layer for
Persistence Diagrams and New Graph Topological Signatures", *Proceedings
of the Twenty Third International Conference on Artificial Intelligence
and Statistics, PMLR 108*, pp. 2786--2796, 2020.
"""

import torch
from torch.nn import Parameter, Linear, init
from torch.nn.functional import relu
from torch.nn.utils.rnn import pad_sequence
from torch_topological.nn import PersistenceInformation


class PermutationEquivariant(torch.nn.Module):
    """A deepset layer.

    This layer provides a permutation equivariant map from the set of
    points in the persistence diagram to a (N,q) tensor where N is the
    number of barcodes.  See [1] for details.

    References
    ----------
    .. [1] M. Zaheer et al., "Deep Sets", *Advances in Neural
    Information Processing Systems 30*, 2017.
    """

    def __init__(self, output_dim, perm_op="max"):
        """Initialize new module.

        Parameters
        ----------
        output_dim : int
            The dimension of the features of the output.

        perm_op : str
            Must be "max", "min" or "sum".  Indicates the permutation
            equivariant operation used in the layer.  Default is "max".
        """
        super().__init__()
        self.weight = Parameter(torch.empty(2, output_dim))
        self.bias = Parameter(torch.empty(output_dim))
        self.gamma = Parameter(torch.empty(2, output_dim))
        self.perm_op = perm_op
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight, a=0., b=1.)
        init.uniform_(self.bias, a=0., b=1.)
        init.uniform_(self.gamma, a=0., b=1.)

    def forward(self, diagram):
        """Implement forward pass.

        Parameters
        ----------
        diagram : Tensor
            Input persistence barcodes.  Diagram is of shape
            (..., N, 2).

        Returns
        -------
        Tensor
            Tensor of shape (..., N, output_dim).
        """
        if self.perm_op is not None:
            if self.perm_op == "max":
                beta = torch.amax(diagram, -2)[..., None, :]
            elif self.perm_op == "min":
                beta = torch.amin(diagram, -2)[..., None, :]
            elif self.perm_op == "sum":
                beta = torch.sum(diagram, -2)[..., None, :]
            else:
                raise Exception("perm_op should be 'min', 'max' or 'sum'.")
            return (torch.matmul(diagram, self.weight)
                    - torch.matmul(beta, self.gamma) + self.bias)
        else:
            return torch.matmul(diagram, self.weight) + self.bias


class Image(torch.nn.Module):
    """A persistence image layer.

    This layer maps persistence diagrams to their persistence images.
    """

    def __init__(self, image_size, image_bnds=((-0.01, 1.01), (-0.01, 1.01))):
        """Initialize new module.

        Parameters
        ----------
        image_size : tuple
            The dimensions of the output image.  Has to be a tuple of
            two positive integers

        image_bnds : tuple
            A tuple of tuples.  Describes the bounds of the axes of the
            image.  Default is ((-0.01, 1.01), (-0.01, 1.01)).
        """
        super().__init__()
        self.sigma = Parameter(torch.empty(image_size[1], image_size[0]))
        self.image_size = image_size
        self.image_bnds = image_bnds
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.sigma, a=0., b=3.)

    def forward(self, diagram):
        """Implement forward pass.

        Parameters
        ----------
        diagram : Tensor
            Input persistence barcodes.  Diagram is of shape
            (..., N, 2).

        Returns
        -------
        Tensor
            Tensor of shape (..., N, image_size[1], image_size[0]).
        """
        diagram[..., 1] = diagram[..., 1] - diagram[..., 0]
        coords = [
            torch.linspace(
                self.image_bnds[i][0],
                self.image_bnds[i][1],
                self.image_size[i]
            ) for i in range(diagram.size(-1))
        ]
        mu = torch.stack(torch.meshgrid(coords, indexing='xy'), dim=0)
        diagram = diagram[..., None, None]
        sum = torch.sum(-0.5 * ((diagram - mu) / self.sigma)**2, dim=-3)
        return (torch.exp(sum) / (2*torch.pi*(self.sigma)**2))


class Landscape(torch.nn.Module):
    """A persistence landscape layer.

    This layer evaluates q points on triangular functions associated to
    each barcode.  See [1] and [2] for details.

    References
    ----------
    .. [1] M. Carriere et al., "PersLay: A Neural Network Layer for
    Persistence Diagrams and New Graph Topological Signatures",
    *Proceedings of the Twenty Third International Conference on
    Artificial Intelligence and Statistics, PMLR 108*, pp. 2786--2796,
    2020.

    .. [2] C. Hofer et al., "Learning Representations of Persistence
    Barcodes", *Journal of Machine Learning Research 20(126)*, pp. 1--
    45, 2019.
    """

    def __init__(self, output_dim):
        """Initialize new module.

        Parameters
        ----------
        output_dim : int
            The dimension of the features of the output, which is the
            number of samples evaluated.
        """
        super().__init__()
        self.samples = Parameter(torch.empty(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.samples, a=0., b=1.)

    def forward(self, diagram):
        """Implement forward pass.

        Parameters
        ----------
        diagram : Tensor
            Input persistence barcodes.  Diagram is of shape
            (..., N, 2).

        Returns
        -------
        Tensor
            Tensor of shape (..., N, output_dim).
        """
        x = diagram[..., 0][..., None]
        y = diagram[..., 1][..., None]
        return relu(torch.minimum(self.samples - x, y - self.samples))


class BettiCurve(torch.nn.Module):
    """A Betti curve layer.

    This layer evaluates q points on smoothened step functions
    associated to each barcode.
    """

    def __init__(self, output_dim, theta=10):
        """Initialize new module.

        Parameters
        ----------
        output_dim : int
            The dimension of the features of the output, which is the
            number of samples evaluated.

        theta : float
            Parameter in the sigmoid function used to approximate
            step functions associated to barcodes.  Default is 10.
        """
        super().__init__()
        self.samples = Parameter(torch.empty(output_dim))
        self.theta = theta
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.samples, a=0., b=1.)

    def forward(self, diagram):
        """Implement forward pass.

        Parameters
        ----------
        diagram : Tensor
            Input persistence barcodes.  Diagram is of shape
            (..., N, 2).

        Returns
        -------
        Tensor
            Tensor of shape (..., N, output_dim).
        """
        x = diagram[..., 0][..., None]
        y = diagram[..., 1][..., None]
        minimum = torch.minimum(self.samples - x, y - self.samples)
        return 1 / (1 + torch.exp(-1 * self.theta * minimum))


class Entropy(torch.nn.Module):
    """A persistence entropy layer.

    This layer evaluates q points on smoothened step functions
    associated to each barcode weighted by the barcode's contribution to
    persistence entropy.
    """

    def __init__(self, output_dim, theta=10):
        """Initialize new module.

        Parameters
        ----------
        output_dim : int
            The dimension of the features of the output, which is the
            number of samples evaluated.

        theta : float
            Parameter in the sigmoid function used to approximate
            step functions associated to barcodes.  Default is 10.
        """
        super().__init__()
        self.samples = Parameter(torch.empty(output_dim))
        self.theta = theta
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.samples, a=0., b=1.)

    def forward(self, diagram):
        """Implement forward pass.

        Parameters
        ----------
        diagram : Tensor
            Input persistence barcodes.  Diagram is of shape
            (..., N, 2).

        Returns
        -------
        Tensor
            Tensor of shape (..., N, output_dim).
        """
        x = diagram[..., 0][..., None]
        y = diagram[..., 1][..., None]
        length = y - x
        prob = length / (torch.sum(length, dim=-2)[..., None, :])
        entropy_terms = torch.where(prob > 0,
                                    -1 * prob * torch.log(prob), prob)
        minimum = torch.minimum(self.samples - x, y - self.samples)
        return entropy_terms / (1 + torch.exp(-1 * self.theta * minimum))


class Exponential(torch.nn.Module):
    """A exponential structural element layer.

    This layer evaluates points in the persistence diagram on q
    exponential structural elements.  See [1] and [2] for details.

    References
    ----------
    .. [1] M. Carriere et al., "PersLay: A Neural Network Layer for
    Persistence Diagrams and New Graph Topological Signatures",
    *Proceedings of the Twenty Third International Conference on
    Artificial Intelligence and Statistics, PMLR 108*, pp. 2786--2796,
    2020.

    .. [2] C. Hofer et al., "Learning Representations of Persistence
    Barcodes", *Journal of Machine Learning Research 20(126)*, pp. 1--
    45, 2019.
    """

    def __init__(self, output_dim):
        """Initialize new module.

        Parameters
        ----------
        output_dim : int
            The dimension of the features of the output, which is the
            number of structural elements.
        """
        super().__init__()
        self.mu = Parameter(torch.empty(2, output_dim))
        self.sigma = Parameter(torch.empty(2, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.mu, a=0., b=1.)
        init.constant_(self.sigma, 3.)

    def forward(self, diagram):
        """Implement forward pass.

        Parameters
        ----------
        diagram : Tensor
            Input persistence barcodes.  Diagram is of shape
            (..., N, 2).

        Returns
        -------
        Tensor
            Tensor of shape (..., N, output_dim).
        """
        sum = torch.sum(((diagram[..., None] - self.mu) * self.sigma)**2,
                        dim=-2)
        return torch.exp(-1 * sum)


class Rational(torch.nn.Module):
    """A rational structural element layer.

    This layer evaluates points in the persistence diagram on q
    rational structural elements.  See [1] for details.

    References
    ----------
    .. [1] C. Hofer et al., "Learning Representations of Persistence
    Barcodes", *Journal of Machine Learning Research 20(126)*, pp. 1--
    45, 2019.
    """

    def __init__(self, output_dim):
        """Initialize new module.

        Parameters
        ----------
        output_dim : int
            The dimension of the features of the output, which is the
            number of structural elements.
        """
        super().__init__()
        self.mu = Parameter(torch.empty(2, output_dim))
        self.sigma = Parameter(torch.empty(2, output_dim))
        self.alpha = Parameter(torch.empty(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.mu, a=0., b=1.)
        init.constant_(self.sigma, 3.)
        init.constant_(self.alpha, 3.)

    def forward(self, diagram):
        """Implement forward pass.

        Parameters
        ----------
        diagram : Tensor
            Input persistence barcodes.  Diagram is of shape
            (..., N, 2).

        Returns
        -------
        Tensor
            Tensor of shape (..., N, output_dim).
        """
        sum = torch.sum(torch.abs((diagram[..., None] - self.mu) * self.sigma),
                        dim=-2)
        return 1 / ((sum + 1)**self.alpha)


class RationalHat(torch.nn.Module):
    """A rational hat structural element layer.

    This layer evaluates points in the persistence diagram on q
    rational hat structural elements.  See [1] for details.

    References
    ----------
    .. [1] C. Hofer et al., "Learning Representations of Persistence
    Barcodes", *Journal of Machine Learning Research 20(126)*, pp. 1--
    45, 2019.
    """

    def __init__(self, output_dim, q=2):
        """Initialize new module.

        Parameters
        ----------
        output_dim : int
            The dimension of the features of the output, which is the
            number of structural elements.

        q : int
            The order of the norm.  Default is 2.
        """
        super().__init__()
        self.mu = Parameter(torch.empty(2, output_dim))
        self.r = Parameter(torch.empty(output_dim))
        self.q = q
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.mu, a=0., b=1.)
        init.constant_(self.r, 3.)

    def forward(self, diagram):
        """Implement forward pass.

        Parameters
        ----------
        diagram : Tensor
            Input persistence barcodes.  Diagram is of shape
            (..., N, 2).

        Returns
        -------
        Tensor
            Tensor of shape (..., N, output_dim).
        """
        norm = torch.norm(diagram[..., None] - self.mu, p=self.q, dim=-2)
        return 1 / (1 + norm) - 1 / (1 + torch.abs(torch.abs(self.r) - norm))


class PersLay(torch.nn.Module):
    """A neural network layer PersLay.

    This layer processes persistence information and maps each barcode
    to q features.  Then it performs a permutation invariant operation
    and summarizes the persistence diagram.  See [1] for details.

    References
    ----------
    .. [1] M. Carriere et al., "PersLay: A Neural Network Layer for
    Persistence Diagrams and New Graph Topological Signatures",
    *Proceedings of the Twenty Third International Conference on
    Artificial Intelligence and Statistics, PMLR 108*, pp. 2786--2796,
    2020.
    """

    def __init__(
            self,
            output_dim,
            point_transform,
            op,
            weight_fn=None,
            dim=0,
            k=None,
            **kwargs
    ):
        """Initialize new module.

        Parameters
        ----------
        output_dim : int or tuple
            The dimension of the features of the output.  When
            point_transform is "Image", then output_dim can be a tuple
            of two positive integers.  If point_transform is "Image" and
            output_dim is an integer, then it assumes that the image
            size to be (output_dim, output_dim).

        point_transform : str or Callable
            The layer that maps persistence barcodes to q dimensional
            features.  If point_transform is a string, it must be
            "PermutationEquivariant", "Image", "Landscape",
            "BettiCurve", "Entropy", "Exponential", "Rational",
            "RationalHat" or "Line".  The first eight initializes the
            corresponding nn.Module, the last one initializes nn.Linear.
            Else point_transform has to be a Callable, which takes in
            tensors of shape (B, N, 2) as input and outputs a tensor of
            shape (B, N, q) if data is batched, else (N, 2) and (N, q)
            respectively.

        op : str
            One of "max", "min", "sum", "mean", "topk", which
            corresponds to the permutation invariant operations max,
            min, sum, mean and kth largest value, that aggregates the
            features across barcodes.

        weight_fn : Callable, optional
            Weights applied to each barcode features before aggregation.
            It should take in tensors of size (B, N, 2) as input and
            outputs a tensor of shape (B, N, 1) if data is batched, else
            shapes should be (N, 2) and (N, 1) respectively.  When
            weight_fn is None, weight 1 is applied to all barcode
            features.  Default is None.

        dim : int
            The dimension of persistence homology considered.  Default
            is 0.

        k : int, optional
            This only matters when op is "topk".  It refers to the k in
            the kth largest value.  Default is None.

        **kwargs
            Additional arguments passed to the point_transform layers
            when point_transform is a string.
        """
        super().__init__()
        self.dim = dim

        if op in ["max", "min", "sum", "mean", "topk"]:
            self.op = op
        else:
            raise ValueError("op should be 'max', 'min', "
                             "'sum', 'mean', 'topk'.")

        if op == "topk":
            if isinstance(k, int):
                if k > 0:
                    self.k = k
                else:
                    raise ValueError("When op is 'topk', "
                                     "k must be a positive integer.")
            else:
                raise TypeError("When op is 'topk', "
                                "k must be a positive integer.")

        self.weight_fn = weight_fn

        if isinstance(point_transform, str):
            if point_transform == "PermutationEquivariant":
                self.point_transform = PermutationEquivariant(output_dim,
                                                              **kwargs)
            elif point_transform == "Image":
                if isinstance(output_dim, int):
                    self.point_transform = Image((output_dim, output_dim),
                                                 **kwargs)
                else:
                    self.point_transform = Image(output_dim, **kwargs)
            elif point_transform == "Landscape":
                self.point_transform = Landscape(output_dim)
            elif point_transform == "BettiCurve":
                self.point_transform = BettiCurve(output_dim, **kwargs)
            elif point_transform == "Entropy":
                self.point_transform = Entropy(output_dim, **kwargs)
            elif point_transform == "Exponential":
                self.point_transform = Exponential(output_dim)
            elif point_transform == "Rational":
                self.point_transform = Rational(output_dim)
            elif point_transform == "RationalHat":
                self.point_transform = RationalHat(output_dim, **kwargs)
            elif point_transform == "Line":
                self.point_transform = Linear(2, output_dim, **kwargs)
            else:
                raise ValueError(
                    "point_transform must be either 'PermutationEquivariant', "
                    "'Image', 'Landscape', 'BettiCurve', 'Entropy', "
                    "'Exponential', 'Rational', 'RationalHat', 'Line' "
                    "or a class with forward method."
                )
        else:
            self.point_transform = point_transform

    def forward(self, x):
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
        x = self.preprocessing(x)
        if self.weight_fn is None:
            x = self.point_transform(x)
        else:
            x = self.weight_fn(x) * self.point_transform(x)
        if isinstance(self.point_transform, Image):
            op_dim = -3
        else:
            op_dim = -2
        if self.op == "max":
            x = torch.amax(x, dim=op_dim)
        elif self.op == "min":
            x = torch.amin(x, dim=op_dim)
        elif self.op == "sum":
            x = torch.sum(x, dim=op_dim)
        elif self.op == "mean":
            x = torch.mean(x, dim=op_dim)
        elif self.op == "topk":
            x = -1 * (torch.kthvalue(-1 * x, self.k, dim=op_dim).values)

        return x

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
