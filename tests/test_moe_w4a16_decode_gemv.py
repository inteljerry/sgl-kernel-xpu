"""moe_w4a16_decode_gemv against an fp64 reference and the existing path.

The decode GEMV must compute what fused_experts(use_mxfp4_w4a16=True,
activation="silu", swiglu_limit=10, routed_scaling_factor=1.5) computes. Both
ops run on the same weight tensors (the existing layout) and are compared with
a float64 reference that has no intermediate rounding
(tests/moe_w4a16_decode_reference.py). Acceptance rule: the new op's error
against the reference is at most 1.25 x the existing op's error + 1e-4, by
max-abs and by Frobenius norm, both relative to the reference RMS; either
metric failing fails the case.

Coverage:
- Shapes: DeepSeek-V4-Flash routed experts per GPU at TP 8 (E 256, topk 6,
  H 4096, I 256), decode batches T = 1, 2, 4, 8; topk 6 / I 512, topk 8 /
  I 256 and topk 8 / I 512 for the other kernel instantiations.
- Routing: independent experts per token, every token on the same experts,
  experts 0 and E - 1, tokens sharing half their experts, and a token that
  lists one expert twice.
- Inputs: N(0, 1) hidden states and N(0, 64) ones (the second drives gate and
  up past the clamp at 10).
- Weights: uniformly random E2M1 codes with E8M0 exponents 115..125, MXFP4
  quantizations of N(0, 0.02) weights (trained-weight code distribution) for
  the active experts, every E2M1 code over narrow and wide exponent ranges.
- Real checkpoint layers (MOE_GEMV_CKPT_DIR, a DeepSeek-V4-Flash-0731
  checkpoint): MOE_GEMV_CKPT_LAYERS (default "0,3,42": the hash-routed
  layer 0, routed through its tid2eid table, and two noaux layers) at TP rank
  MOE_GEMV_TP_RANK (default 0).
- Recorded decode activations (MOE_GEMV_ACTS_DIR, recorded at the MoE input
  of a DeepSeek-V4-Flash server): hidden states, topk ids and weights of real
  decode steps early, middle and late in a 512-token generation, at contexts
  of about 1k, 8k and 32k and batches of 1, 2, 4, 8 with mixed lengths,
  replayed against their checkpoint layer (needs MOE_GEMV_CKPT_DIR).
- Prefill: not applicable; the op serves T * topk <= 64 routes only.

Set MOE_GEMV_REPORT to a file path to append one JSON line per case.

    pytest tests/test_moe_w4a16_decode_gemv.py -v -s
"""

import functools
import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe_w4a16_decode_reference import build_mxfp4_experts  # noqa: E402
from moe_w4a16_decode_reference import error_metrics  # noqa: E402
from moe_w4a16_decode_reference import make_routing  # noqa: E402
from moe_w4a16_decode_reference import moe_decode_reference_fp64  # noqa: E402
from moe_w4a16_decode_reference import within_gate  # noqa: E402
from sgl_kernel import fused_experts, moe_w4a16_decode_gemv  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")

# DeepSeek-V4-Flash routed experts, per GPU at TP 8.
E, H, I, TOPK = 256, 4096, 256, 6
RSF, LIMIT = 1.5, 10.0
NUM_HASH_LAYERS = 3
TOKENS = [1, 2, 4, 8]
KINDS = ["random", "shared", "edges", "mixed", "dup"]
DEVICE = "xpu"
CKPT_DIR = os.environ.get("MOE_GEMV_CKPT_DIR")
CKPT_LAYERS = [
    int(v)
    for v in os.environ.get("MOE_GEMV_CKPT_LAYERS", "0,3,42").split(",")
    if v.strip()
]
CKPT_RANK = int(os.environ.get("MOE_GEMV_TP_RANK", "0"))
ACTS_DIR = os.environ.get("MOE_GEMV_ACTS_DIR")


def _cases():
    cases = [(t, kind, 1.0) for kind in KINDS for t in TOKENS]
    cases += [(t, kind, 8.0) for kind in ("random", "shared") for t in TOKENS]
    return [(t, kind, scale, 10 * t + i) for i, (t, kind, scale) in enumerate(cases)]


def _to_device(w13, s13, w2, s2):
    # The dtypes of fused_experts: int8 packed weights, float8_e8m0fnu scales.
    return (
        w13.view(torch.int8).to(DEVICE),
        s13.view(torch.float8_e8m0fnu).to(DEVICE),
        w2.view(torch.int8).to(DEVICE),
        s2.view(torch.float8_e8m0fnu).to(DEVICE),
    )


@pytest.fixture(scope="module")
def dsv4_weights():
    active = set()
    for t, kind, _, seed in _cases():
        ids, _ = make_routing(t, E, TOPK, kind, seed)
        active.update(ids.view(-1).tolist())
    active = sorted(active)
    w13, s13 = build_mxfp4_experts(E, 2 * I, H, realistic_experts=active, seed=1)
    w2, s2 = build_mxfp4_experts(E, H, I, realistic_experts=active, seed=2)
    return (w13, s13, w2, s2), _to_device(w13, s13, w2, s2)


def _hidden(num_tokens, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (
        (torch.randn((num_tokens, H), generator=g) * scale)
        .to(torch.bfloat16)
        .to(DEVICE)
    )


def _existing(x, dev, topk_weights, topk_ids, rsf=RSF, limit=LIMIT):
    w13, s13, w2, s2 = dev
    return fused_experts(
        x,
        w13,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        use_mxfp4_w4a16=True,
        w1_scale=s13,
        w2_scale=s2,
        routed_scaling_factor=rsf,
        swiglu_limit=limit,
    )


def _gemv(x, dev, topk_weights, topk_ids, rsf=RSF, limit=LIMIT):
    w13, s13, w2, s2 = dev
    return moe_w4a16_decode_gemv(
        x,
        w13,
        w2,
        topk_weights,
        topk_ids,
        s13,
        s2,
        routed_scaling_factor=rsf,
        swiglu_limit=limit,
    )


def _record(case, **values):
    path = os.environ.get("MOE_GEMV_REPORT")
    line = json.dumps({"case": case, **values})
    print(line)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def _check_against_reference(
    case, x, cpu, dev, topk_weights, topk_ids, rsf=RSF, limit=LIMIT
):
    out_new = _gemv(x, dev, topk_weights, topk_ids, rsf, limit)
    out_existing = _existing(x, dev, topk_weights, topk_ids, rsf, limit)
    torch.xpu.synchronize()
    assert out_new.shape == x.shape and out_new.dtype == torch.bfloat16
    assert torch.isfinite(out_new).all()

    ref = moe_decode_reference_fp64(x, *cpu, topk_weights, topk_ids, rsf, limit)
    e_new = error_metrics(out_new, ref)
    e_existing = error_metrics(out_existing, ref)
    diff = (out_new.float() - out_existing.float()).abs()
    ok = within_gate(e_new, e_existing)
    _record(
        case,
        new=e_new,
        existing=e_existing,
        ratio_max_abs=e_new["max_abs"] / max(e_existing["max_abs"], 1e-300),
        ratio_frobenius=e_new["frobenius"] / max(e_existing["frobenius"], 1e-300),
        bitwise_equal_fraction=(out_new == out_existing).float().mean().item(),
        max_abs_new_vs_existing=diff.max().item(),
        within_gate=ok,
    )
    assert ok, f"{case}: new {e_new} exceeds 1.25 x existing {e_existing} + 1e-4"
    return out_new


@pytest.mark.parametrize("num_tokens,kind,x_scale,seed", _cases())
def test_error_within_existing_error(dsv4_weights, num_tokens, kind, x_scale, seed):
    cpu, dev = dsv4_weights
    ids, tw = make_routing(num_tokens, E, TOPK, kind, seed)
    x = _hidden(num_tokens, seed, x_scale)
    _check_against_reference(
        f"dsv4-tp8 T={num_tokens} {kind} x*{x_scale}",
        x,
        cpu,
        dev,
        tw.to(DEVICE),
        ids.to(DEVICE),
    )


@pytest.mark.parametrize("exp_range", [(115, 126), (100, 151)])
@pytest.mark.parametrize("num_tokens", [1, 8])
def test_all_codes_and_scale_ranges(num_tokens, exp_range):
    # Uniform codes: every E2M1 code, including the 0.5 subnormal and -0, in
    # every group; E8M0 exponents over a narrow and a wide range.
    w13, s13 = build_mxfp4_experts(E, 2 * I, H, seed=3)
    w2, s2 = build_mxfp4_experts(E, H, I, seed=4)
    g = torch.Generator().manual_seed(5)
    s13 = torch.randint(*exp_range, s13.shape, dtype=torch.uint8, generator=g)
    s2 = torch.randint(*exp_range, s2.shape, dtype=torch.uint8, generator=g)
    cpu = (w13, s13, w2, s2)
    ids, tw = make_routing(num_tokens, E, TOPK, "random", 6)
    x = _hidden(num_tokens, 7)
    _check_against_reference(
        f"uniform-codes exp={exp_range} T={num_tokens}",
        x,
        cpu,
        _to_device(*cpu),
        tw.to(DEVICE),
        ids.to(DEVICE),
    )


@pytest.mark.parametrize("topk,inter", [(6, 512), (8, 256), (8, 512)])
@pytest.mark.parametrize("num_tokens", [1, 4, 8])
def test_topk_and_intermediate_sizes(num_tokens, topk, inter):
    # The kernel instantiations other than topk 6 / I 256: one w2 row per sub-group
    # message at I = 512, two at I = 256. T = 8 at topk 8 is 64 routes.
    e = 64
    ids, tw = make_routing(num_tokens, e, topk, "random", 8)
    active = sorted(set(ids.view(-1).tolist()))
    w13, s13 = build_mxfp4_experts(e, 2 * inter, H, realistic_experts=active, seed=9)
    w2, s2 = build_mxfp4_experts(e, H, inter, realistic_experts=active, seed=10)
    cpu = (w13, s13, w2, s2)
    x = _hidden(num_tokens, 11)
    _check_against_reference(
        f"topk{topk}-I{inter} T={num_tokens}",
        x,
        cpu,
        _to_device(*cpu),
        tw.to(DEVICE),
        ids.to(DEVICE),
    )


def test_index_and_weight_dtypes(dsv4_weights):
    _, dev = dsv4_weights
    ids, tw = make_routing(8, E, TOPK, "random", 12)
    x = _hidden(8, 12)
    base = _gemv(x, dev, tw.to(DEVICE), ids.to(DEVICE))
    out_i64 = _gemv(x, dev, tw.to(DEVICE), ids.to(torch.int64).to(DEVICE))
    tw_bf16 = tw.to(torch.bfloat16)
    out_bf16 = _gemv(x, dev, tw_bf16.to(DEVICE), ids.to(DEVICE))
    out_bf16_as_f32 = _gemv(x, dev, tw_bf16.float().to(DEVICE), ids.to(DEVICE))
    assert torch.equal(base, out_i64)
    assert torch.equal(out_bf16, out_bf16_as_f32)


def test_int64_ids_outside_the_experts_add_nothing(dsv4_weights):
    # An int64 id outside [0, E) adds nothing, also when its low 32 bits name a
    # valid expert: both kernels check the id before they narrow it to int.
    _, dev = dsv4_weights
    ids, tw = make_routing(2, E, TOPK, "random", 21)
    x = _hidden(2, 21)
    expert = int(ids[0, 1])
    skipped = ids.to(torch.int64).clone()
    skipped[0, 1] = -1
    aliased = skipped.clone()
    aliased[0, 1] = (1 << 32) + expert
    full = _gemv(x, dev, tw.to(DEVICE), ids.to(DEVICE))
    out_skipped = _gemv(x, dev, tw.to(DEVICE), skipped.to(DEVICE))
    out_aliased = _gemv(x, dev, tw.to(DEVICE), aliased.to(DEVICE))
    assert not torch.equal(full[0], out_skipped[0])
    assert torch.equal(out_skipped, out_aliased)


def test_token_output_independent_of_batch(dsv4_weights):
    # A token's output depends only on its own inputs: the same token alone and
    # inside a batch that shares its experts gives the same bits.
    _, dev = dsv4_weights
    ids, tw = make_routing(8, E, TOPK, "mixed", 17)
    x = _hidden(8, 17)
    batch = _gemv(x, dev, tw.to(DEVICE), ids.to(DEVICE))
    for t in (0, 3, 7):
        alone = _gemv(
            x[t : t + 1], dev, tw[t : t + 1].to(DEVICE), ids[t : t + 1].to(DEVICE)
        )
        assert torch.equal(alone[0], batch[t]), t


def test_deterministic_and_graph_replay(dsv4_weights):
    _, dev = dsv4_weights
    num_tokens = 8
    ids, tw = make_routing(num_tokens, E, TOPK, "random", 13)
    x = _hidden(num_tokens, 13)
    ids, tw = ids.to(DEVICE), tw.to(DEVICE)
    first = _gemv(x, dev, tw, ids)
    second = _gemv(x, dev, tw, ids)
    assert torch.equal(first, second)

    static_x, static_ids, static_tw = x.clone(), ids.clone(), tw.clone()
    _gemv(static_x, dev, static_tw, static_ids)  # warm the workspace outside capture
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(xpu_graph=graph):
        static_out = _gemv(static_x, dev, static_tw, static_ids)

    # Each replay sees new routing and activations; any state left on the device
    # by an earlier replay would show up as a mismatch against the eager call.
    for seed, kind in (
        (14, "random"),
        (15, "shared"),
        (16, "edges"),
        (18, "dup"),
        (14, "random"),
    ):
        new_ids, new_tw = make_routing(num_tokens, E, TOPK, kind, seed)
        new_x = _hidden(num_tokens, seed)
        static_x.copy_(new_x)
        static_ids.copy_(new_ids.to(DEVICE))
        static_tw.copy_(new_tw.to(DEVICE))
        graph.replay()
        eager = _gemv(new_x, dev, new_tw.to(DEVICE), new_ids.to(DEVICE))
        torch.xpu.synchronize()
        assert torch.equal(
            static_out, eager
        ), f"graph replay differs for {kind} seed {seed}"


# --- Real checkpoint layers and recorded decode activations -------------------


class _Checkpoint:
    def __init__(self, model_dir):
        from safetensors import safe_open

        self.model_dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.handles = {}
        self.safe_open = safe_open

    def name(self, name):
        return name if name in self.weight_map else f"model.{name}"

    def get(self, name):
        name = self.name(name)
        path = os.path.join(self.model_dir, self.weight_map[name])
        if path not in self.handles:
            self.handles[path] = self.safe_open(path, framework="pt")
        return self.handles[path].get_slice(name)

    def has(self, name):
        return self.name(name) in self.weight_map


@functools.lru_cache(maxsize=1)
def _checkpoint():
    return _Checkpoint(CKPT_DIR)


@functools.lru_cache(maxsize=1)
def _checkpoint_layer(layer, tp_rank, tp_size=8):
    """One layer's routed experts, TP-sharded like the sglang loader: w1 and w3
    rows [r*I, (r+1)*I) stacked as gate then up, w2 input columns of rank r.
    Returns the CPU tensors and their device copies."""
    ck = _checkpoint()
    inter = 2048 // tp_size
    r0 = tp_rank * inter
    w13 = torch.empty((E, 2 * inter, H // 2), dtype=torch.uint8)
    s13 = torch.empty((E, 2 * inter, H // 32), dtype=torch.uint8)
    w2 = torch.empty((E, H, inter // 2), dtype=torch.uint8)
    s2 = torch.empty((E, H, inter // 32), dtype=torch.uint8)
    for e in range(E):
        prefix = f"layers.{layer}.ffn.experts.{e}"
        for half, proj in enumerate(("w1", "w3")):
            rows = slice(half * inter, (half + 1) * inter)
            w13[e, rows] = ck.get(f"{prefix}.{proj}.weight")[r0 : r0 + inter].view(
                torch.uint8
            )
            s13[e, rows] = ck.get(f"{prefix}.{proj}.scale")[r0 : r0 + inter].view(
                torch.uint8
            )
        w2[e] = ck.get(f"{prefix}.w2.weight")[:, r0 // 2 : (r0 + inter) // 2].view(
            torch.uint8
        )
        s2[e] = ck.get(f"{prefix}.w2.scale")[:, r0 // 32 : (r0 + inter) // 32].view(
            torch.uint8
        )
    cpu = (w13, s13, w2, s2)
    return cpu, _to_device(*cpu)


def _hash_routing(layer, num_tokens, seed):
    """Routing of a hash layer: the tid2eid rows of random token ids (the last
    token repeats the first), with random normalized weights."""
    tid2eid = _checkpoint().get(f"layers.{layer}.ffn.gate.tid2eid")[:]
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, tid2eid.shape[0], (num_tokens,), generator=g)
    if num_tokens > 1:
        tokens[-1] = tokens[0]
    ids = tid2eid[tokens].to(torch.int32)
    w = torch.rand(ids.shape, generator=g) + 0.05
    return ids, (w / w.sum(dim=-1, keepdim=True)).to(torch.float32)


def _checkpoint_cases():
    cases = []
    for layer in CKPT_LAYERS:
        kinds = ["hash"] if layer < NUM_HASH_LAYERS else []
        kinds += ["random", "shared", "mixed"]
        for kind in kinds:
            for t in TOKENS:
                for scale in (1.0, 8.0):
                    cases.append((layer, kind, t, scale))
    return cases


@pytest.mark.skipif(not CKPT_DIR, reason="MOE_GEMV_CKPT_DIR not set")
@pytest.mark.parametrize("layer,kind,num_tokens,x_scale", _checkpoint_cases())
def test_real_checkpoint_layer(layer, kind, num_tokens, x_scale):
    cpu, dev = _checkpoint_layer(layer, CKPT_RANK)
    seed = 1000 * layer + 10 * num_tokens + int(x_scale)
    if kind == "hash":
        ids, tw = _hash_routing(layer, num_tokens, seed)
    else:
        ids, tw = make_routing(num_tokens, E, TOPK, kind, seed)
    x = _hidden(num_tokens, seed, x_scale)
    _check_against_reference(
        f"checkpoint layer={layer} rank={CKPT_RANK} {kind} T={num_tokens} x*{x_scale}",
        x,
        cpu,
        dev,
        tw.to(DEVICE),
        ids.to(DEVICE),
    )


def _recorded_files():
    if not ACTS_DIR:
        return []
    files = sorted(Path(ACTS_DIR).glob("*/L*-s*-r*.pt"))
    # Group by layer and rank so the one-layer weight cache is reused.
    return sorted(
        files, key=lambda p: (p.name.split("-")[0], p.name.split("-")[-1], str(p))
    )


@pytest.mark.skipif(
    not (CKPT_DIR and ACTS_DIR),
    reason="MOE_GEMV_CKPT_DIR and MOE_GEMV_ACTS_DIR not set",
)
@pytest.mark.parametrize(
    "path", _recorded_files(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_recorded_decode_activations(path):
    rec = torch.load(path, map_location="cpu", weights_only=True)
    assert rec["activation"] == "silu"
    cpu, dev = _checkpoint_layer(rec["layer_id"], rec["tp_rank"])
    rsf = (
        1.0
        if rec["routed_scaling_factor"] is None
        else float(rec["routed_scaling_factor"])
    )
    limit = float(rec["swiglu_limit"])
    _check_against_reference(
        f"recorded {rec['tag']} layer={rec['layer_id']} step={rec['step']} T={rec['x'].shape[0]}",
        rec["x"].to(DEVICE),
        cpu,
        dev,
        rec["topk_weights"].to(DEVICE),
        rec["topk_ids"].to(DEVICE),
        rsf,
        limit,
    )


def test_recorded_activations_present():
    # When a recording is named, it must hold all three batch compositions.
    if not ACTS_DIR:
        pytest.skip("MOE_GEMV_ACTS_DIR not set")
    batches = {
        torch.load(p, map_location="cpu", weights_only=True)["x"].shape[0]
        for p in _recorded_files()
    }
    assert {1, 2, 4, 8} <= batches, f"recorded batch sizes {sorted(batches)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
