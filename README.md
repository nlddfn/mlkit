# ML-Kit: a wrapper around scikit-learn and more
## Getting Started

* `make test`

* `make install`


## The idea:
ML-kit is a simple wrapper around scikit-learn and related libraries to simplify my job and hopefully reduce the boilerplate code in a typical ML project. The library includes a selection of Supervised and Unsupervisd models that I found useful/explored in the past.

## The pipeline:
ML-kit extends the idea of PIPE operator explored in scikit-learn. The library allows to create a chain of trasformations (Data In , Data Out) and models (Data In, Model Out). Models can be combined in stacks and ensembles.
