![Coverage](coverage.svg) ![Interrogate](interrogate.svg) ![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)

# Recommender systems toolkit

Simple and lightweight toolkit for building and deploying recommender systems.
## Installation
```sh
pip install rstk
```

## Usage
### Building a recommender system
```sh
rstk build <algorithm> <dataset> [model-path]
```

### Serving it as an HTTP endpoint
```
rstk run <model/path>
```
Runs it on the default port `11235`

For further information dive into the [examples](https://github.com/rat-nick/rstk/tree/main/examples).
