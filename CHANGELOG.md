# Changelog

`pytorch-topological` follows the [semantic versioning](https://semver.org).
This changelog contains all notable changes in the project.

# v0.1.4

## Added

- Batch handler for point cloud complexes
- Sliced Wasserstein distance kernel
- Support for pre-computed distances in Vietoris--Rips complex

## Fixed

- Device compatibility issue (tensor being created on the wrong device)
- Use of `dim` flag for alpha complexes
- Various documentation issues and coding style problems
