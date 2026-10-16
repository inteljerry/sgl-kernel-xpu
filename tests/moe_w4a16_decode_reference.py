"""fp64 reference and input builders for the decode MXFP4 W4A16 MoE.

Used by tests/test_moe_w4a16_decode_gemv.py and
benchmark/bench_moe_w4a16_decode_gemv.py. Imports only torch, so the
reference itself can be checked on a machine without an XPU.

The reference computes the routed-expert MoE of fused_experts(
use_mxfp4_w4a16=True, activation="silu", swiglu_limit=L) in float64 with no
intermediate rounding:

    h      = x[t] @ W13[e]^T            gate = h[:I], up = h[I:]
    a      = silu(min(gate, L)) * clamp(up, -L, L)
    out[t] = sum_k (a_k @ W2[e_k]^T) * w_k * routed_scaling_factor

W13 and W2 are the exact dequantized MXFP4 weights: E2M1 value times
2^(e - 127) for the E8M0 byte e of each group of 32 inputs.

Error metrics follow the numerics gate: both are relative to the RMS of the
reference, max_abs = max |out - ref| / rms(ref) and frobenius = ||out - ref|| /
||ref||; a kernel passes when each metric is at most 1.25 x the existing
kernel's value + 1e-4.
"""

import torch

MXFP4_GROUP = 32
GATE_RATIO = 1.25
GATE_SLACK = 1e-4

# E2M1 codes 0..15: sign in bit 3, magnitude index in bits 0..2.
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _codes(packed: torch.Tensor) -> torch.Tensor:
    p = packed.view(torch.uint8).cpu().to(torch.int64)
    return torch.stack([p & 0xF, p >> 4], dim=-1).reshape(p.shape[0], -1)


def dequant_mxfp4_fp64(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[N, K/2] packed E2M1 (low nibble is the even element) and [N, K/32]
    E8M0 bytes -> [N, K] float64 on the CPU."""
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float64)[_codes(packed)]
    e = scale.view(torch.uint8).cpu().to(torch.int64)
    s = torch.ldexp(torch.ones_like(e, dtype=torch.float64), e - 127)
    return values * s.repeat_interleave(MXFP4_GROUP, dim=1)


def moe_decode_reference_fp64(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: float,
    swiglu_limit: float,
) -> torch.Tensor:
    """float64 [T, H] on the CPU. Dequantizes only the experts in topk_ids."""
    x = hidden_states.detach().cpu().double()
    ids = topk_ids.detach().cpu().to(torch.int64)
    weights = topk_weights.detach().cpu().double()
    num_tokens, topk = ids.shape
    inter = w13.shape[1] // 2
    out = torch.zeros((num_tokens, x.shape[1]), dtype=torch.float64)
    for e in ids.unique().tolist():
        w13_e = dequant_mxfp4_fp64(w13[e], w13_scale[e])
        w2_e = dequant_mxfp4_fp64(w2[e], w2_scale[e])
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ w13_e.t()
        gate = h[:, :inter].clamp(max=swiglu_limit)
        up = h[:, inter:].clamp(-swiglu_limit, swiglu_limit)
        act = gate * torch.sigmoid(gate) * up
        y = act @ w2_e.t()
        out.index_add_(
            0, tok, y * (weights[tok, slot] * routed_scaling_factor)[:, None]
        )
    return out


def _dequant_existing_bf16(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # The existing kernel converts E2M1 to bf16 and multiplies by the scale read as
    # float(e << 23), rounding the product to bf16 (exact for normal ranges).
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32)[_codes(packed)]
    e = scale.view(torch.uint8).cpu().to(torch.int32)
    s = (e << 23).view(torch.float32).repeat_interleave(MXFP4_GROUP, dim=1)
    return (values * s).to(torch.bfloat16)


def moe_decode_existing_emulation(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: float,
    swiglu_limit: float,
    return_intermediate: bool = False,
):
    """bf16 [T, H] on the CPU with the existing path's rounding points: bf16
    GEMM1 output, bf16 clamp and activation, bf16 GEMM2 output per route, fp32
    combine in topk order. fp32 matmul accumulation stands in for the DPAS
    kernel, so this approximates the error of fused_experts where the XPU op
    cannot run.
    With return_intermediate, also returns the activated GEMM1 output per route
    (bf16 [T * topk, I], row t * topk + k).
    """
    x = hidden_states.detach().cpu().to(torch.bfloat16).float()
    ids = topk_ids.detach().cpu().to(torch.int64)
    weights = topk_weights.detach().cpu().float()
    num_tokens, topk = ids.shape
    inter = w13.shape[1] // 2
    limit = float(torch.tensor(swiglu_limit, dtype=torch.bfloat16))
    y = torch.zeros((num_tokens, topk, x.shape[1]), dtype=torch.bfloat16)
    acts = torch.zeros((num_tokens, topk, inter), dtype=torch.bfloat16)
    for e in ids.unique().tolist():
        w13_e = _dequant_existing_bf16(w13[e], w13_scale[e]).float()
        w2_e = _dequant_existing_bf16(w2[e], w2_scale[e]).float()
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = (x[tok] @ w13_e.t()).to(torch.bfloat16).float()
        gate = h[:, :inter].clamp(max=limit)
        up = h[:, inter:].clamp(-limit, limit)
        act = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
        acts[tok, slot] = act
        y[tok, slot] = (act.float() @ w2_e.t()).to(torch.bfloat16)
    acc = torch.zeros((num_tokens, x.shape[1]), dtype=torch.float32)
    for k in range(topk):
        acc += y[:, k].float() * weights[:, k : k + 1] * routed_scaling_factor
    if return_intermediate:
        return acc.to(torch.bfloat16), acts.reshape(num_tokens * topk, inter)
    return acc.to(torch.bfloat16)


def error_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Max-abs and Frobenius error of out against ref, both relative to the
    reference RMS (see the module docstring)."""
    diff = out.detach().cpu().double() - ref
    rms = ref.pow(2).mean().sqrt().clamp_min(1e-300)
    return {
        "max_abs": (diff.abs().max() / rms).item(),
        "frobenius": (diff.norm() / ref.norm().clamp_min(1e-300)).item(),
    }


def within_gate(new: dict, existing: dict) -> bool:
    return all(
        new[k] <= GATE_RATIO * existing[k] + GATE_SLACK
        for k in ("max_abs", "frobenius")
    )


def build_mxfp4_experts(
    num_experts: int,
    rows: int,
    cols: int,
    realistic_experts=(),
    std: float = 0.02,
    seed: int = 0,
):
    """Packed [E, rows, cols/2] uint8 codes and [E, rows, cols/32] uint8 E8M0
    scales on the CPU.

    Every expert gets uniformly random codes (all 16 E2M1 codes appear) with
    exponents 115..125, which is the range of real routed-expert scales. The
    experts in realistic_experts are replaced by MXFP4 quantizations of
    N(0, std) weights (the round-to-nearest-even OCP MX recipe), whose code
    distribution matches trained weights.
    """
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(
        0, 256, (num_experts, rows, cols // 2), dtype=torch.uint8, generator=g
    )
    scales = torch.randint(
        115,
        126,
        (num_experts, rows, cols // MXFP4_GROUP),
        dtype=torch.uint8,
        generator=g,
    )
    for e in realistic_experts:
        w = torch.randn((rows, cols), generator=g, dtype=torch.float32) * std
        p, s = quantize_mxfp4_rne(w)
        packed[e].copy_(p)
        scales[e].copy_(s)
    return packed, scales


def quantize_mxfp4_rne(w: torch.Tensor):
    """OCP MX MXFP4 quantization of a [rows, cols] float tensor: shared exponent
    floor(log2(amax)) - 2 per 32 values, E2M1 round-to-nearest-even with
    saturation at 6. Returns packed codes and E8M0 bytes."""
    rows, cols = w.shape
    blocks = w.double().reshape(rows, cols // MXFP4_GROUP, MXFP4_GROUP)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    shared = (
        torch.floor(
            torch.log2(torch.where(amax > 0, amax, torch.full_like(amax, 2.0**-126)))
        )
        - 2
    )
    shared = shared.clamp(-127, 127)
    scaled = blocks / torch.ldexp(torch.ones_like(shared), shared.to(torch.int64))
    mag = scaled.abs().clamp(max=6.0)
    # Midpoints between the E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6. bucketize
    # puts a value equal to a midpoint in the lower bucket; ties go to the code
    # with an even mantissa bit (index 0, 2, 4, 6), so an odd lower bucket moves up.
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float64)
    idx = torch.bucketize(mag, mids)
    tie = (idx < 7) & (mag == mids[idx.clamp(max=6)]) & (idx % 2 == 1)
    idx = idx + tie.to(torch.int64)
    codes = (idx | ((scaled < 0) & (idx > 0)).to(torch.int64) << 3).reshape(rows, cols)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    e8m0 = (shared.reshape(rows, cols // MXFP4_GROUP) + 127).to(torch.uint8)
    return packed, e8m0


ROUTING_KINDS = ("random", "shared", "edges", "mixed", "dup")


def make_routing(
    num_tokens: int, num_experts: int, topk: int, kind: str, seed: int = 0
):
    """topk_ids int32 [T, topk] and normalized float32 topk_weights on the CPU.

    Every token picks topk distinct experts, as the router does, except for
    "dup". kind: "random" (independent across tokens), "shared" (every token
    picks the same experts, so each active expert serves all T tokens, as when
    a batch repeats one token in a hash-routed layer), "edges" (experts 0 and
    num_experts - 1 always active, the rest random), "mixed" (token t reuses
    the last topk / 2 experts of token t - 1, so experts serve one or two
    tokens), "dup" (the last slot of every token repeats its first expert; a
    hash table row with a repeated expert routes this way).
    """
    assert kind in ROUTING_KINDS, kind
    g = torch.Generator().manual_seed(seed)
    if kind == "shared":
        base = torch.randperm(num_experts, generator=g)[:topk]
        ids = base.repeat(num_tokens, 1)
    else:
        ids = torch.stack(
            [torch.randperm(num_experts, generator=g)[:topk] for _ in range(num_tokens)]
        )
        if kind == "edges":
            for t in range(num_tokens):
                rest = [e for e in ids[t].tolist() if e not in (0, num_experts - 1)]
                ids[t] = torch.tensor([0, num_experts - 1] + rest[: topk - 2])
        elif kind == "mixed":
            half = topk // 2
            for t in range(1, num_tokens):
                keep = ids[t - 1, topk - half :].tolist()
                fresh = [
                    e
                    for e in torch.randperm(num_experts, generator=g).tolist()
                    if e not in keep
                ]
                ids[t] = torch.tensor(keep + fresh[: topk - half])
        elif kind == "dup":
            ids[:, topk - 1] = ids[:, 0]
    w = torch.rand((num_tokens, topk), generator=g) + 0.05
    w = w / w.sum(dim=-1, keepdim=True)
    return ids.to(torch.int32), w.to(torch.float32)
