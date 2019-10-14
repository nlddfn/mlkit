# ML Kit: a wrapper around scikit-learn
## Getting Started
Clone the repo and set a virtual environment. Tox expects Python 3.7 nevertheless the code is tested also in python 3.6. Should `make test` fail, please change tox.ini to account for your version of python.

* `make test`

* `make install`


## The idea:
ML-kit is a simple wrapper around scikit-learn and related libraries to simplify my job and hopefully reduce the boilerplate code in a typical ML project. The library includes a selection of Supervised and Unsupervisd models that I found useful/explored in the past.

## The pipeline:
MLkit extends the idea of PIPE operator explored in scikit-learn. The library allows to create a chain of trasformations (data in => data out) and models (data in => model out). Models can be combined in stacks and ensembles.
