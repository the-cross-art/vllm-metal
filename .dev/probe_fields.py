"""Find which multimodal field is not a tensor, and what it actually is.

Runs vLLM's own multimodal processing path without starting an engine.
Debug only; never part of the PR.
"""

import vllm.multimodal.inputs as mmi
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

CKPT = "/Users/imran.nazir/lfm2-vl-test"

# Report instead of asserting, so every field is visible, not just the first
# one that fails.
_orig = mmi.MultiModalFlatField.build_elems


def _traced(self, modality, key, data):
    nested = not mmi.is_list_of(self.slices, slice, check="all")
    print(
        f"  {key:22} type={type(data).__name__:12} nested_slices={nested} "
        f"n_slices={len(self.slices)} shape={getattr(data, 'shape', None)}"
    )
    return _orig(self, modality, key, data)


mmi.MultiModalFlatField.build_elems = _traced

mc = ModelConfig(model=CKPT, limit_mm_per_prompt={"image": 1})
proc = MULTIMODAL_REGISTRY.create_processor(mc)

print("fields seen:")
try:
    # Exactly the call the engine makes in MultiModalBudget.
    out = proc.get_dummy_mm_inputs({"image": 1})
    print("get_dummy_mm_inputs() OK")
    for mod in out["mm_kwargs"]:
        for index, item in enumerate(out["mm_kwargs"][mod]):
            for key in item:
                data = item[key].data
                shape = getattr(data, "shape", None)
                print(f"  {mod}[{index}].{key:18} {type(data).__name__} {shape}")
except Exception as exc:
    print(f"get_dummy_mm_inputs() FAILED: {type(exc).__name__}: {exc}")
