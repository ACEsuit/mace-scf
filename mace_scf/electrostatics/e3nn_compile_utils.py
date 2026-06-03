from typing import Any

import e3nn
from e3nn import o3

from e3nn.o3._linear import _codegen_linear
from e3nn.o3._tensor_product._tensor_product import (
    codegen_tensor_product_left_right,
    codegen_tensor_product_right,
)


def _first_if_tuple(value: Any) -> Any:
    if isinstance(value, tuple):
        return value[0]
    return value


def replace_e3nn_script_codegen_with_fx(module) -> None:
    """Replace e3nn TorchScript codegen children with FX GraphModules in-place.

    e3nn 0.4.x stores generated Linear/TensorProduct implementations as
    RecursiveScriptModules by default. Dynamo cannot trace those reliably,
    especially inside the fixed-step SCF loop. Regenerating the same e3nn
    codegen with jit_script_fx=False preserves weights and numerical behavior
    while leaving Dynamo with FX GraphModules to trace.
    """
    old_defaults = e3nn.get_optimization_defaults().copy()
    e3nn.set_optimization_defaults(jit_script_fx=False)
    try:
        _replace_e3nn_script_codegen_with_fx(module)
    finally:
        e3nn.set_optimization_defaults(**old_defaults)


def _replace_e3nn_script_codegen_with_fx(module) -> None:
    for child in module.children():
        _replace_e3nn_script_codegen_with_fx(child)

    defaults = e3nn.get_optimization_defaults()
    if isinstance(module, o3.Linear):
        graph_module, _, _ = _codegen_linear(
            module.irreps_in,
            module.irreps_out,
            module.instructions,
            None,
            None,
            shared_weights=module.shared_weights,
            optimize_einsums=getattr(
                module,
                "_optimize_einsums",
                defaults["optimize_einsums"],
            ),
        )
        module._codegen_register({"_compiled_main": graph_module})
        return

    if isinstance(module, o3.TensorProduct):
        graph_module_left_right = _first_if_tuple(
            codegen_tensor_product_left_right(
                module.irreps_in1,
                module.irreps_in2,
                module.irreps_out,
                module.instructions,
                module.shared_weights,
                defaults["specialized_code"],
                getattr(
                    module,
                    "_optimize_einsums",
                    defaults["optimize_einsums"],
                ),
            )
        )
        graph_module_right = _first_if_tuple(
            codegen_tensor_product_right(
                module.irreps_in1,
                module.irreps_in2,
                module.irreps_out,
                module.instructions,
                module.shared_weights,
                defaults["specialized_code"],
                getattr(
                    module,
                    "_optimize_einsums",
                    defaults["optimize_einsums"],
                ),
            )
        )
        module._codegen_register(
            {
                "_compiled_main_left_right": graph_module_left_right,
                "_compiled_main_right": graph_module_right,
            }
        )
