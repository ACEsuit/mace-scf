"""Dispatch contract for `make_model_wrapper`.

`make_model_wrapper` selects a wrapper class by `model.__class__.__name__` and
has one construction call per branch. Nothing else in the test suite touches it,
so a branch that is missing a constructor argument stays latent until a real
training run picks that model. That is exactly how `LocalSourcesModelWrapper`
and `QEqModelWrapper` ended up being constructed without `model=`.

Dispatch is on the class *name* only, so these tests use stand-in modules rather
than real models: the point is to reach every branch cheaply, not to run a
forward pass.
"""

import inspect

import pytest
import torch

from mace_scf.electrostatics.fixed_point_options import (
    validate_fixed_point_training_options,
)
from mace_scf.utils.model_training_wrappers import (
    DefaultModelWrapper,
    FixedPointWrapper,
    LocalSourcesModelWrapper,
    QEqModelWrapper,
    make_model_wrapper,
)


OUTPUT_ARGS = {
    "energy": True,
    "forces": True,
    "virials": False,
    "stress": False,
    "polarizability": False,
}

# Every model class name `make_model_wrapper` claims to support, and the wrapper
# it must return. Keep in sync with the dispatch branches.
DISPATCH_CASES = [
    ("MACE", DefaultModelWrapper),
    ("ScaleShiftMACE", DefaultModelWrapper),
    ("FixedPoint", FixedPointWrapper),
    ("FixedPointCore", FixedPointWrapper),
    ("LocalCharges", LocalSourcesModelWrapper),
    ("LocalSplitCharges", LocalSourcesModelWrapper),
    ("FixedChargeBaselinedMACE", LocalSourcesModelWrapper),
    ("MACEQEq", QEqModelWrapper),
]


class _StandInModel(torch.nn.Module):
    """Minimal module; only its class name matters to the dispatcher."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def _model_named(class_name: str) -> torch.nn.Module:
    return type(class_name, (_StandInModel,), {})()


def _build(class_name: str):
    model = _model_named(class_name)
    wrapper = make_model_wrapper(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        output_args=OUTPUT_ARGS,
        fixed_point_training_options=validate_fixed_point_training_options(
            {"mode": "direct"}
        ),
    )
    return model, wrapper


def _call_implementation(wrapper):
    """The function a wrapper call actually lands in.

    Most wrappers define `forward`; `QEqModelWrapper` still overrides `__call__`
    directly. Resolve on the class so both are introspectable.
    """
    cls = type(wrapper)
    if "forward" in vars(cls):
        return cls.forward
    return cls.__call__


@pytest.mark.parametrize(
    "class_name,expected_wrapper", DISPATCH_CASES, ids=[c for c, _ in DISPATCH_CASES]
)
def test_dispatch_constructs_and_binds_model(class_name, expected_wrapper):
    """Every branch constructs, returns the right wrapper, and binds the model.

    Regression test for the two branches that omitted `model=` and raised
    `TypeError: __init__() missing 1 required positional argument: 'model'`.
    """
    model, wrapper = _build(class_name)

    assert isinstance(wrapper, expected_wrapper)
    assert wrapper.model is model


@pytest.mark.parametrize("class_name", [c for c, _ in DISPATCH_CASES])
def test_dispatch_registers_model_as_submodule(class_name):
    """The bound model is a registered child, not a plain attribute.

    Assigning a Module before `super().__init__()` raises, so this also pins the
    `super().__init__()` call that `QEqModelWrapper` was missing its parens on.
    """
    model, wrapper = _build(class_name)

    assert isinstance(wrapper, torch.nn.Module)
    assert dict(wrapper.named_children()).get("model") is model
    # Reachable through the wrapper, which is what lets DDP see the parameters.
    assert list(wrapper.parameters()) == list(model.parameters())


@pytest.mark.parametrize("class_name", [c for c, _ in DISPATCH_CASES])
def test_wrapper_call_takes_no_model_argument(class_name):
    """Wrappers own their model; callers pass only the batch.

    Pins the `wrapper(model, batch_dict, ...)` -> `wrapper(batch_dict, ...)`
    migration so a wrapper cannot quietly reacquire a `model` parameter.
    """
    _, wrapper = _build(class_name)
    params = list(inspect.signature(_call_implementation(wrapper)).parameters)

    assert params[0] == "self"
    assert params[1] == "batch_dict"
    assert "model" not in params


def test_unknown_model_class_raises():
    model = _model_named("NotAModelWeSupport")

    with pytest.raises(ValueError, match="does not have a wrapper class"):
        make_model_wrapper(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            output_args=OUTPUT_ARGS,
        )


def test_fixed_point_requires_training_options():
    model = _model_named("FixedPoint")

    with pytest.raises(ValueError, match="fixed_point_training_options"):
        make_model_wrapper(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            output_args=OUTPUT_ARGS,
        )
