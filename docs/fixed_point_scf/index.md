# Fixed-Point SCF Models

Fixed-point SCF models solve a self-consistent electrostatic density problem.
They build a local density contribution, compute long-range field features from
a trial density, update the density, and repeat until convergence or until the
configured SCF step limit is reached.

## Pages

Its reccomended to read these in order:

- [Model information](model.md)
- [Using fixed-point SCF models](using.md)
- [Training](training.md)
- [Implicit differentiation](implicit_diff.md)
- [Evaluation](evaluation.md)
- [ASE calculator](calculator.md)