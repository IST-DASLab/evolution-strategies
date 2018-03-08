# Evolution Strategies

This is a pytorch implementation of ES, as described
[here](https://blog.openai.com/evolution-strategies/).

`base.py` contains all ES stuff.

`regression.py` is example usage where we learn a polynomial of some degree. It is based on
[the pytorch example](https://github.com/pytorch/examples/tree/master/regression).
See `python regression.py --help` for options on running. The MNIST example uses the same
arguments.

`mnist.py` is an attempt at learning MNIST using ES. It is extremely slow, and doesn't really work
well, although this is probably my (martins) fault ¯\\_(ツ)_/¯.

## Setup

Install pytorch from [here](http://pytorch.org).

```
pip install -r requirements.txt
```
