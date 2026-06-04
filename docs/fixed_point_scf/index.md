# Fixed-Point SCF Models

Fixed-point SCF models solve a self-consistent electrostatic density problem.
They build a local density contribution, compute long-range field features from
a trial density, update the density, and repeat until convergence or until the
configured SCF step limit is reached.

## Pages

Its reccomended to read these in order:

```{toctree}
:maxdepth: 2

model
using
training
implicit_diff
evaluation
calculator
```
