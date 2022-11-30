# Changelog

`pytorch-topological` follows the [semantic versioning](https://semver.org).
This changelog contains all notable changes in the project.

# v0.1.6

## Fixed

- Fixed various documentation typos
- Fixed bug in `make_tensor` creation for single batches

# v0.1.5

## Added

- A bunch of new test cases
- Alpha complex class (`AlphaComplex`)
- Dimension selector class (`SelectByDimension`)
- Discussing additional packages in documentation
- Linting for pull requests

## Fixed

- Improved contribution guidelines
- Improved documentation of summary statistics loss
- Improved overall maintainability
- Improved test cases
- Simplified multi-scale kernel usage (distance calculations with different exponents)
- Test case for cubical complexes
- Usage of seed parameter for shape generation (following `numpy guidelines`)

# v0.1.4

## Added

- Batch handler for point cloud complexes
- Sliced Wasserstein distance kernel
- Support for pre-computed distances in Vietoris--Rips complex

## Fixed

- Device compatibility issue (tensor being created on the wrong device)
- Use of `dim` flag for alpha complexes
- Various documentation issues and coding style problems
