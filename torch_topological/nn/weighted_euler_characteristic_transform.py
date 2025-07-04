"""Weighted Euler Characteristic Transform(WECT) implementation"""

from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


class EulerDistance(nn.Module):
    """Calculate the L2 norm between two (Weighted)Euler Curves/Transforms"""

    def __init__(self):
        super().__init__()

    def forward(self, ec1, ec2):
        mse = torch.nn.MSELoss(reduction="mean")
        return mse(ec1.float(), ec2.float())


class WeightedEulerCurve(nn.Module):
    """ "Calculate Weighted Euler Characteristic Transform of a given 3D tensor

        This is an implementation of the WECT, following [Jiang_2020]_.

    References
    ----------
    .. [Jiang_2020] Q. Jiang et al., "The Weighted Euler Curve Transform for
       Shape and Image Analysis", Proceedings of the IEEE/CVF Conference on
       Computer Vision and Pattern Recognition (CVPR) Workshops, 2020,
       pp. 844-845
    """

    def __init__(self, num_directions=100, num_steps=30, prod=False):
        """Create new WECT module.

        Parameters
        ----------
        num_directions : int
            Specifies the number of random directions to be sampled for
            computation of the WECT.

        num_steps : int
            Number of steps to be used for the approximation of a single curve.

        prod : bool (default=False)
            Specifies whether to use the product of constituent vertices of an
            edge/square/cube will be used as the value of the edge/square/cube
            or if the maximum of the constituent vertices will be used.
        """
        super().__init__()
        torch.manual_seed(4)
        np.random.seed(4)
        self.prod = prod
        self.num_directions = num_directions
        self.num_steps = num_steps
        self.directions = torch.stack(
            [self._obtainDirection() for _ in range(num_directions)], dim=0
        )

    def forward(self, x):
        """Calculate the Weighted Euler Characteristic Transform(WECT)
        for an input 3D float tensor.

        Parameters
        ----------
        x : 3D float torch tensor

        Returns
        -------
        torch.tensor
            A 3D tensor of dimension (num_directions, num_steps, 1) which is a
            stacked tensor of num_directions Weighted Euler Curves of X,
            each of which is a 1D tensor of length num_steps."""

        num_directions, num_steps = self.num_directions, self.num_steps
        self.sz = x.shape
        vertices = self._genVertices(x)
        vertIndices = self._getVertexIndex(vertices)
        vertVals = torch.take(x, vertIndices.long())

        edges, edgeVals = self._genEdges(x)
        squares, sqVals = self._genSquares(x)
        cubes, cubeVals = self._genCubes(x)

        vertices, edges, squares, cubes = (
            vertices.to(x.device),
            edges.to(x.device),
            squares.to(x.device),
            cubes.to(x.device),
        )

        min_height = -(3 ** (1 / 2)) * x.shape[0]
        max_height = -min_height
        dh = (max_height - min_height) / num_steps
        euler_curves = []
        self.directions = self.directions.to(x.device)
        for placeholder in range(num_directions):
            dir_vector = self.directions[placeholder]
            vert_heights, edge_heights, square_heights, cube_heights = (
                vertices,
                edges,
                squares,
                cubes,
            )

            vertValSorted, edgeValSorted, squareValSorted, cubeValSorted = (
                None,
                None,
                None,
                None,
            )

            if vertices.shape != torch.tensor([]).shape:
                vert_heights = torch.matmul(
                    dir_vector.float(), vertices.float()
                )
                vert_heights, idx = torch.sort(vert_heights)
                vertValSorted = vertVals[idx]

            if edges.shape != torch.tensor([]).shape:
                edge_heights, _ = torch.max(
                    torch.stack(
                        [
                            torch.matmul(dir_vector.float(), edges[i].float())
                            for i in range(2)
                        ],
                        axis=1,
                    ),
                    dim=1,
                )

                edge_heights, idx = torch.sort(edge_heights)
                edgeValSorted = edgeVals[idx]

            if squares.shape != torch.tensor([]).shape:
                square_heights, _ = torch.max(
                    torch.stack(
                        [
                            torch.matmul(
                                dir_vector.float(), squares[i].float()
                            )
                            for i in range(4)
                        ],
                        axis=1,
                    ),
                    dim=1,
                )
                square_heights, idx = torch.sort(square_heights)
                squareValSorted = sqVals[idx]

            if cubes.shape != torch.tensor([]).shape:
                cube_heights, _ = torch.max(
                    torch.stack(
                        [
                            torch.matmul(dir_vector.float(), cubes[i].float())
                            for i in range(8)
                        ],
                        axis=1,
                    ),
                    dim=1,
                )

                cube_heights, idx = torch.sort(cube_heights)
                cubeValSorted = cubeVals[idx]

            if vertValSorted is not None:
                vertValSorted = torch.cumsum(vertValSorted, dim=0)
                vertValSorted = torch.cat(
                    (torch.tensor([0]).to(x.device), vertValSorted)
                )

            if edgeValSorted is not None:
                edgeValSorted = torch.cumsum(edgeValSorted, dim=0)
                edgeValSorted = torch.cat(
                    (torch.tensor([0]).to(x.device), edgeValSorted)
                )

            if squareValSorted is not None:
                squareValSorted = torch.cumsum(squareValSorted, dim=0)
                squareValSorted = torch.cat(
                    (torch.tensor([0]).to(x.device), squareValSorted)
                )

            if cubeValSorted is not None:
                cubeValSorted = torch.cumsum(cubeValSorted, dim=0)
                cubeValSorted = torch.cat(
                    (torch.tensor([0]).to(x.device), cubeValSorted)
                )

            inf = 1e9
            vert_heights = torch.cat(
                (torch.tensor([-inf]).to(x.device), vert_heights))
            edge_heights = torch.cat(
                (torch.tensor([-inf]).to(x.device), edge_heights))
            square_heights = torch.cat(
                (torch.tensor([-inf]).to(x.device), square_heights))
            cube_heights = torch.cat(
                (torch.tensor([-inf]).to(x.device), cube_heights))

            intervals = torch.arange(min_height, max_height + dh, dh).to(
                x.device)

            v_ptr1 = (
                torch.searchsorted(vert_heights, intervals, right=True) - 1)
            e_ptr1 = (
                torch.searchsorted(edge_heights, intervals, right=True) - 1)
            s_ptr1 = (
                torch.searchsorted(square_heights, intervals, right=True) - 1)
            c_ptr1 = (
                torch.searchsorted(cube_heights, intervals, right=True) - 1)

            v_Val = 0 if vertValSorted is None else vertValSorted[v_ptr1]
            e_Val = 0 if edgeValSorted is None else edgeValSorted[e_ptr1]
            s_Val = 0 if squareValSorted is None else squareValSorted[s_ptr1]
            c_Val = 0 if cubeValSorted is None else cubeValSorted[c_ptr1]

            euler_curves.append(v_Val - e_Val + s_Val - c_Val)
        return torch.stack(euler_curves, dim=0)

    def _obtainDirection(self):
        """
        https://gist.github.com/andrewbolster/10274979
        Generates a random 3D unit vector (direction) with a uniform
        spherical distribution. Algo from:
        http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
        """
        phi = np.random.uniform(0, np.pi * 2)
        costheta = np.random.uniform(-1, 1)

        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return torch.tensor([x, y, z], dtype=torch.double)

    def _genSquares(self, x):
        """Obtains coordinates and values of the adjacent squares in the 3D
        tensor. Input: x - 3D tensor"""
        device = x.device
        arr = torch.clone(x)
        arr = arr.unsqueeze(0).float()
        arr[arr > 0] = 1

        weights = (
            torch.tensor([[1, 1], [1, 1]])
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )
        arrH = torch.transpose(arr, 0, 1)
        arrW = torch.transpose(arrH, 0, 2)
        arrB = torch.transpose(arrW, 0, 3)

        # Performing convolution operation to efficiently obtain all
        # nonzero indices of squares
        convH = F.conv2d(arrH, weights).squeeze()
        convW = F.conv2d(arrW, weights).squeeze()
        convB = F.conv2d(arrB, weights).squeeze()

        idxH = torch.nonzero(convH == 4)
        idxW = torch.nonzero(convW == 4)
        idxW = torch.index_select(idxW, 1, torch.tensor([1, 0, 2]).to(device))
        idxB = torch.nonzero(convB == 4)
        idxB = torch.index_select(idxB, 1, torch.tensor([1, 2, 0]).to(device))

        coordsH = torch.stack(
            [
                idxH + torch.tensor([0, i, j]).to(device)
                for i in range(2)
                for j in range(2)
            ],
            dim=1,
        )

        coordsW = torch.stack(
            [
                idxW + torch.tensor([i, 0, j]).to(device)
                for i in range(2)
                for j in range(2)
            ],
            dim=1,
        )

        coordsB = torch.stack(
            [
                idxB + torch.tensor([i, j, 0]).to(device)
                for i in range(2)
                for j in range(2)
            ],
            dim=1,
        )

        coords = torch.cat([coordsH, coordsW, coordsB])
        coords = torch.transpose(coords, 0, 1)
        coords = torch.transpose(coords, 1, 2)
        sqIndices = self._getVertexIndex(coords[0])
        sqVals1 = torch.take(x, sqIndices.long())
        sqIndices = self._getVertexIndex(coords[1])
        sqVals2 = torch.take(x, sqIndices.long())
        sqIndices = self._getVertexIndex(coords[2])
        sqVals3 = torch.take(x, sqIndices.long())
        sqIndices = self._getVertexIndex(coords[3])
        sqVals4 = torch.take(x, sqIndices.long())
        if self.prod is True:
            sqVals = sqVals1 * sqVals2 * sqVals3 * sqVals4
        else:
            sqVals = torch.maximum(sqVals1, sqVals2)
            sqVals2 = torch.maximum(sqVals3, sqVals4)
            sqVals = torch.maximum(sqVals, sqVals2)
        return coords.float(), sqVals

    def _genCubes(self, x):
        """Obtains coordinates and values of the adjacent cubes in the 3D
        tensor. Input: x - 3D tensor"""
        device = x.device
        x_ = torch.clone(x)
        x_ = x_.unsqueeze(0).unsqueeze(0).float()
        x_[x_ > 0] = 1
        weights = (
            torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        # Performing convolution operation to efficiently obtain all
        # nonzero cubes
        conv3d = F.conv3d(x_, weights).squeeze()
        indices = torch.nonzero(conv3d == 8)
        if indices.numel() == 0:
            return torch.tensor([]), torch.tensor([])

        cubes = torch.stack(
            [
                indices + torch.tensor([i, j, k]).to(device)
                for i in range(2)
                for j in range(2)
                for k in range(2)
            ],
            dim=1
        )

        cubes = cubes.transpose(0, 1).transpose(1, 2).float()
        cubeIndices = self._getVertexIndex(cubes[0])
        cubeVals1 = torch.take(x, cubeIndices.long())
        cubeIndices = self._getVertexIndex(cubes[1])
        cubeVals2 = torch.take(x, cubeIndices.long())
        cubeIndices = self._getVertexIndex(cubes[2])
        cubeVals3 = torch.take(x, cubeIndices.long())
        cubeIndices = self._getVertexIndex(cubes[3])
        cubeVals4 = torch.take(x, cubeIndices.long())
        cubeIndices = self._getVertexIndex(cubes[4])
        cubeVals5 = torch.take(x, cubeIndices.long())
        cubeIndices = self._getVertexIndex(cubes[5])
        cubeVals6 = torch.take(x, cubeIndices.long())
        cubeIndices = self._getVertexIndex(cubes[6])
        cubeVals7 = torch.take(x, cubeIndices.long())
        cubeIndices = self._getVertexIndex(cubes[7])
        cubeVals8 = torch.take(x, cubeIndices.long())

        if self.prod is True:
            cubeVals = (
                cubeVals1
                * cubeVals2
                * cubeVals3
                * cubeVals4
                * cubeVals5
                * cubeVals6
                * cubeVals7
                * cubeVals8
            )
        else:
            cubeVals = torch.maximum(cubeVals1, cubeVals2)
            cubeVals2 = torch.maximum(cubeVals3, cubeVals4)
            cubeVals3 = torch.maximum(cubeVals5, cubeVals6)
            cubeVals4 = torch.maximum(cubeVals7, cubeVals8)
            cubeVals = torch.maximum(cubeVals, cubeVals2)
            cubeVals2 = torch.maximum(cubeVals3, cubeVals4)
            cubeVals = torch.maximum(cubeVals, cubeVals2)
        return cubes, cubeVals

    def _genVertices(self, x):
        """Obtains coordinates and values of the vertices in the 3D tensor
        Input: x - 3D tensor"""
        return torch.transpose(torch.nonzero(x), 0, 1).float()

    def _genEdges(self, x):
        """Obtains coordinates and values of the adjacent edges in the 3D
        tensor. Input: x - 3D tensor"""
        device = x.device
        arr = torch.clone(x)
        arr = arr.unsqueeze(0).float()
        arr[arr > 0] = 1
        weights = (torch.tensor([[1, 1]]).unsqueeze(0).unsqueeze(0)
                   .float().to(device))

        weights_ = (
            torch.tensor([[1], [1]])
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        arrH = torch.transpose(arr, 0, 1)
        arrW = torch.transpose(arrH, 0, 2)

        # Performing convolution operation to efficiently obtain all
        # nonzero edges
        convH = F.conv2d(arrH, weights).squeeze()
        convH_ = F.conv2d(arrH, weights_).squeeze()
        convW = F.conv2d(arrW, weights_).squeeze()

        idxH = torch.nonzero(convH == 2)
        idxH_ = torch.nonzero(convH_ == 2)
        idxW = torch.nonzero(convW == 2)
        idxW = torch.index_select(idxW, 1, torch.tensor([1, 0, 2]).to(device))

        coordsH = torch.stack(
            [idxH + torch.tensor([0, 0, i]).to(device) for i in range(2)],
            dim=1)

        coordsH_ = torch.stack(
            [idxH_ + torch.tensor([0, i, 0]).to(device) for i in range(2)],
            dim=1)

        coordsW = torch.stack(
            [idxW + torch.tensor([i, 0, 0]).to(device) for i in range(2)],
            dim=1)

        coords = torch.cat([coordsH, coordsH_, coordsW])
        coords = torch.transpose(coords, 0, 1)
        coords = torch.transpose(coords, 1, 2)
        edgeIndices = self._getVertexIndex(coords[0])
        edgeVals1 = torch.take(x, edgeIndices.long())
        edgeIndices = self._getVertexIndex(coords[1])
        edgeVals2 = torch.take(x, edgeIndices.long())
        if self.prod:
            edgeVals = edgeVals1 * edgeVals2
        else:
            edgeVals = torch.maximum(edgeVals1, edgeVals2)
        return coords.float(), edgeVals

    # Get the flattened index of the vertex, given its coordinates
    def _getVertexIndex(self, coords):
        return (
            coords[2] + self.sz[2] * coords[1]
            + self.sz[1] * self.sz[2] * coords[0])
