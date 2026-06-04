# Implicit Differentiation For Fixed-Point SCF

This page describes the two SCF training modes that avoid differentiating
through every iteration of the fixed-point loop:

- `implicit`
- `linearize_solve`

Both methods first converge the SCF loop without storing the full iteration
graph. They then recover the derivative of the converged solution from a
linearized fixed-point equation.

`linearize_solve` is the recommended method. It uses only PyTorch autograd and
linear algebra, requiring no extra packages.

## Implicit Differentiation

TorchOpt has a good introduction to implicit differentiation [here](https://torchopt.readthedocs.io/en/latest/implicit_diff/implicit_diff.html).

The fixed-point model defines a density update

$$
\mathbf p = F_\theta(\mathbf p, \mu; \mathbf x),
$$

where:

- $\mathbf p$ is the vector of atom-centred multipole coefficients;
- $\mu$ is the Fermi level;
- $\mathbf x$ denotes positions, atomic numbers, cell, external field, and
  other non-SCF inputs;
- $\theta$ denotes model parameters.

Define the residual

$$
R(\mathbf p, \mu; \theta, \mathbf x) =
\mathbf p - F_\theta(\mathbf p, \mu; \mathbf x).
$$

At convergence, the residual is zero, possibly together with an additional
charge constraint. The energy and other observables are then evaluated from
the converged density:

$$
E(\theta, \mathbf x) =
\mathcal E(\mathbf p^\ast, \mu^\ast; \theta, \mathbf x).
$$

Implicit differentiation avoids backpropagating through the SCF iterations.
Instead, it differentiates the equation that defines the converged state.

### Constant Fermi Level

Suppose that for a given training example, the model has been iterated to self consistency. This gives a set of multipoles $\mathbf p^\ast$.
In constant-Fermi mode, $\mu$ is fixed. The only implicit variable is
$\mathbf p^\ast$, which satisfies

$$
R(\mathbf p^\ast; \mu, \theta, \mathbf x) =
\mathbf p^\ast -
F_\theta(\mathbf p^\ast, \mu; \mathbf x) =
0.
$$

For any outer variable $\alpha$, such as a parameter or coordinate,
differentiating the residual gives

$$
\frac{\partial R}{\partial \mathbf p}
\frac{\partial \mathbf p^\ast}{\partial \alpha}
{}+
\frac{\partial R}{\partial \alpha}
= 0,
$$

so

$$
\frac{\partial \mathbf p^\ast}{\partial \alpha}
=
-
\left(
\frac{\partial R}{\partial \mathbf p}
\right)^{-1}
\frac{\partial R}{\partial \alpha}.
$$

The matrix being solved with is

$$
\frac{\partial R}{\partial \mathbf p}
=
I - \frac{\partial F_\theta}{\partial \mathbf p}.
$$

Hence, to compute the gradient $\partial \mathbf p^\ast / \partial \alpha$, one first converges the SCF loop, then computes $\partial F_\theta / \partial \mathbf{p}$ with autograd, then solves the above matrix equations.

`implicit` mode delegates this implicit-gradient calculation to `torchopt`.

## Constant Total Charge

In constant-charge mode, the total charge is fixed and the Fermi level becomes
an implicit variable. The unknown is the augmented vector

$$
\mathbf y =
\begin{bmatrix}
\mathbf p \\
\mu
\end{bmatrix},
$$

The residual has two parts:

$$
\begin{aligned}
R_\mathrm{aug}(\mathbf p, \mu)
&=
\begin{bmatrix}
\mathbf p - F_\theta(\mathbf p, \mu; \mathbf x) \\
Q(\mathbf p) - Q_\mathrm{target}
\end{bmatrix}
\\
&= 0.
\end{aligned}
$$

Here $Q(\mathbf p)$ is the per-graph total charge obtained from the charge
component of the multipole density. In block form, the Jacobian is

$$
\frac{\partial R_\mathrm{aug}}{\partial(\mathbf p,\mu)} =
\begin{bmatrix}
I - \dfrac{\partial F_\theta}{\partial \mathbf p}
&
-\dfrac{\partial F_\theta}{\partial \mu}
\\
\dfrac{\partial Q}{\partial \mathbf p}
&
0
\end{bmatrix}.
$$

Implicit differentiation solves linear systems involving this augmented
Jacobian. This is the same mathematical object used by the current
constant-charge `implicit` implementation.

## Linearize and Solve

`linearize_solve` serves the same purpose as implicit differentiation, but does so in what turns out to be a simpler and more efficient way.

Like in implicit differentation, during training the model is first iterated to convergence on a given training example. In implicit differentiation, we use some matrix algebra to compute the gradient of that solution with respect to upstream parameters. In `linearize_solve` we instead linearize the map around the solution, and then solve for the exact solution in a differentiable way.

The inspiration comes from quadratic QEq-like models. In that case, one has a quadratic enrergy functional $E = E_0 + \mathbf{a}^T \mathbf{p} + \frac{1}{2}\mathbf{p}^T M \mathbf{p}$, and it can be minimized by setting the derivative equall to zero: $- \mathbf{a} = M \mathbf{p}^\ast$, and solving for $\mathbf{p}^\ast$. This doesn't require any iteration and in practice one can then differentiate through the matrix solve quite easily for forces and gradients. 

We can apply the same trick by first converging the model and treating the output as the approximate solution $\mathbf{p}^\text{ref}$, where 'ref' stands for 'reference'. At this point:
$$
R(\mathbf{p}^\text{ref}; \mu, \theta, \mathbf{x}) \approx \mathbf{0}
$$
This also works with $R(\mathbf{p}^\text{ref}) = \mathbf{0}$, but is easier to follow by first imagining an approximate solution. Treating $\mathbf{p}^\text{ref}$ as a fixed constant, one can write the exact solution as 
$$
\mathbf{p}^\ast = \mathbf{p}^\text{ref} + \boldsymbol{\Delta}
$$
Because the reference is very close to the solution, we will have $\boldsymbol{\Delta} \approx 0$. Now, the solution satisfies:
$$
R(\mathbf{p}^\text{ref} + \boldsymbol{\Delta}; \mu, \theta, \mathbf{x}) = R(\mathbf{p}^\text{ref}) + \frac{\partial R}{\partial \mathbf{p}} \boldsymbol\Delta = \mathbf{0}
$$
Hence we can solve for $\boldsymbol\Delta$ by computing the Jacobian $\partial R / \partial \mathbf{p}$ and solving for $\Delta$. We will find that $\boldsymbol\Delta \approx 0$, but if the Jacobian construction and inversion is differentiable, then $\Delta$ will carry the right derivatives with respect to parameters and positions.

The linearize and solve method therefore works by: (i) converge the model to the solution $\mathbf{p}^\text{ref}$ and detach it from the graph, (ii) compute the jacobian and residual and solve for $\Delta$, (iii) set the computed charges as $\mathbf{p}^\ast = \mathbf{p}^\text{ref} + \boldsymbol{\Delta}$ and then use this as a differentiable object in downstream opertations.

Like in `implicit`, there is a slight modification for constant charge mode. 
