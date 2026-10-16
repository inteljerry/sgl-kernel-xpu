"""mhc_fused_post_pre_small_t against the fp64 reference and the existing op.

Production shapes: DeepSeek-V4-Flash at TP 8 runs the fused op at T = 1, 2, 4, 8
(decode graph batch sizes), hidden 4096, fn [24, 16384], with the RMSNorm weight.

The error gate rejects when the new kernel's error against fp64 exceeds 1.25x
the existing kernel's error + 1e-4, by max-abs or by Frobenius norm, both
relative to the reference RMS.

test_chain_mhc_pre_to_hc_post runs one decode step's mHC chain: the existing
mhc_pre at the layer 0 attention boundary, all 85 fused calls (layer 0 FFN, then
attention and FFN of layers 1 to 42), and the existing hc_post after the last
layer, with a fixed elementwise stand-in for each sublayer. The error gate runs
on every fused call with the chain's activations. The final hc_post output is a
composition check: after 85 calls it moves with the reduction order alone (in the
CPU emulation two orders differed by 1.34x in max-abs at T = 1, where a few
per-token mixes carry the error), so the new chain's error against an fp64 chain
is bounded by 2x the largest error of the existing op run with n_splits 0, 16 and 32.

Environment:
- SGL_MHC_CKPT_DIR=<checkpoint dir with config.json and
  model.safetensors.index.json>: the chain uses the real mHC weights of every
  layer and site; without it each site gets its own random weights.
- SGL_MHC_SEEDS: random input draws per shape in test_error_vs_fp64_reference
  (default 8).
- SGL_MHC_ACTS=<file> (format in mhc_reference.load_recorded_inputs): runs the
  gate on activations recorded at the model's call sites, with each record's
  layer weights when SGL_MHC_CKPT_DIR is also set.
"""

import functools
import os

import pytest
import torch
from mhc_reference import (
    HC_MULT,
    NUM_LAYERS,
    cancelling_inputs,
    checkpoint_num_layers,
    error_gate,
    error_stats,
    format_gate_rows,
    fused_call_sites,
    hc_post_ref_fp64,
    hc_pre_ref_fp64,
    load_checkpoint_layer,
    load_recorded_inputs,
    make_inputs,
    make_weights,
    mhc_fused_post_pre_ref_fp64,
)
from sgl_kernel import hc_post, mhc_fused_post_pre, mhc_fused_post_pre_small_t, mhc_pre

DEVICE = "xpu:0"
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 20
NORM_EPS = 1e-6
DECODE_TOKENS = [1, 2, 4, 8]
HIDDEN = 4096
# Reduction orders of the existing op that calibrate the chain composition check.
CHAIN_EXISTING_SPLITS = (0, 16, 32)


@pytest.fixture(autouse=True)
def skip_if_no_xpu():
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")


def _run(op, x, residual, post, comb, fn, hc_scale, hc_base, norm_weight):
    return op(
        x,
        residual,
        post,
        comb,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS if norm_weight is not None else None,
    )


def _ref(x, residual, post, comb, fn, hc_scale, hc_base, norm_weight, streams=None):
    args = [
        v.cpu() if v is not None else None
        for v in (x, residual, post, comb, fn, hc_scale, hc_base, norm_weight)
    ]
    return mhc_fused_post_pre_ref_fp64(
        *args[:7],
        rms_eps=RMS_EPS,
        hc_pre_eps=HC_PRE_EPS,
        hc_sinkhorn_eps=HC_SINKHORN_EPS,
        hc_post_mult_value=HC_POST_MULT_VALUE,
        sinkhorn_repeat=SINKHORN_REPEAT,
        norm_weight=args[7],
        norm_eps=NORM_EPS if norm_weight is not None else None,
        streams=None if streams is None else streams.cpu(),
    )


def _check_shapes(out, t, d):
    residual_cur, post_cur, comb_cur, layer_input = out
    assert (
        residual_cur.shape == (t, HC_MULT, d) and residual_cur.dtype == torch.bfloat16
    )
    assert post_cur.shape == (t, HC_MULT, 1) and post_cur.dtype == torch.float32
    assert comb_cur.shape == (t, HC_MULT, HC_MULT) and comb_cur.dtype == torch.float32
    assert layer_input.shape == (t, d) and layer_input.dtype == torch.bfloat16


@pytest.mark.parametrize("d", [4096, 7168])
@pytest.mark.parametrize("t", DECODE_TOKENS)
@pytest.mark.parametrize("with_norm", [False, True])
def test_matches_existing(t, d, with_norm):
    inputs = make_inputs(t, d, DEVICE, seed=t, with_norm=with_norm)
    new = _run(mhc_fused_post_pre_small_t, *inputs)
    existing = _run(mhc_fused_post_pre, *inputs)
    _check_shapes(new, t, d)

    tols = [(1e-2, 1e-2), (2e-2, 2e-2), (2e-2, 2e-2), (2e-2, 2e-2)]
    for name, a, b, (atol, rtol) in zip(
        ("residual_cur", "post", "comb", "layer_input"), new, existing, tols
    ):
        torch.testing.assert_close(
            a,
            b,
            atol=atol,
            rtol=rtol,
            msg=f"{name} mismatch (T={t}, D={d}, norm={with_norm})",
        )


def _gate_cases(cases, header):
    """The error gate over a list of input tuples, pooled per output: the new
    kernel's error against fp64 is at most 1.25x the existing kernel's + 1e-4, by
    max-abs and by Frobenius norm (both relative to the reference RMS)."""
    new_out, existing_out, refs = (
        [[] for _ in range(4)],
        [[] for _ in range(4)],
        [[] for _ in range(4)],
    )
    # Diagnostic, not gating: pre part referenced from each kernel's own bf16 streams.
    new_cond, existing_cond = [[] for _ in range(4)], [[] for _ in range(4)]
    residual_equal = 0
    residual_total = 0
    for inputs in cases:
        new = _run(mhc_fused_post_pre_small_t, *inputs)
        existing = _run(mhc_fused_post_pre, *inputs)
        ref = _ref(*inputs)
        ref_new = _ref(*inputs, streams=new[0])
        ref_existing = _ref(*inputs, streams=existing[0])
        residual_equal += (new[0] == existing[0]).sum().item()
        residual_total += new[0].numel()
        for i in range(4):
            new_out[i].append(new[i].cpu())
            existing_out[i].append(existing[i].cpu())
            refs[i].append(ref[i])
            new_cond[i].append(ref_new[i])
            existing_cond[i].append(ref_existing[i])

    passed, rows = error_gate(
        error_stats(new_out, refs), error_stats(existing_out, refs)
    )
    _, cond_rows = error_gate(
        error_stats(new_out, new_cond), error_stats(existing_out, existing_cond)
    )

    print(f"\n{header} cases={len(cases)}")
    print(f"residual_cur bitwise equal to existing: {residual_equal}/{residual_total}")
    print(format_gate_rows(rows))
    print("diagnostic, pre part against each kernel's own streams:")
    print(format_gate_rows(cond_rows[2:]))
    assert (
        passed
    ), "new kernel error exceeds 1.25x the existing kernel error + 1e-4 against fp64"


@pytest.mark.parametrize("t", DECODE_TOKENS)
def test_error_vs_fp64_reference(t):
    n_seeds = int(os.environ.get("SGL_MHC_SEEDS", "8"))
    cases = [
        make_inputs(t, HIDDEN, DEVICE, seed=1000 + seed) for seed in range(n_seeds)
    ]
    _gate_cases(cases, f"T={t} D={HIDDEN} random weights, seeds={n_seeds}")


@pytest.mark.parametrize("cancel", [1e-1, 1e-2, 1e-3])
def test_cancelling_streams_vs_fp64(cancel):
    """The gate on streams that cancel under the pre weights, where the kernel
    sums ||y||^2 directly instead of from the Gram matrix. The seeds are those of
    the CPU emulation test, which checks that these inputs take the direct sum."""
    cases = [
        tuple(
            None if v is None else v.to(DEVICE)
            for v in cancelling_inputs(4, HIDDEN, cancel, seed=300 + seed)
        )
        for seed in range(3)
    ]
    _gate_cases(cases, f"T=4 D={HIDDEN} cancelling streams, cancel={cancel}")


_SITE_WEIGHTS = {}


def _site_weights():
    """{(layer, site): (fn, hc_scale, hc_base, norm_weight)} for every pre-mix site
    of one decode step, including the layer 0 attention site that mhc_pre serves.
    Real checkpoint weights with SGL_MHC_CKPT_DIR, else per-site random weights."""
    ckpt = os.environ.get("SGL_MHC_CKPT_DIR")
    key = ckpt or "random"
    if key not in _SITE_WEIGHTS:
        num_layers = checkpoint_num_layers(ckpt) if ckpt else NUM_LAYERS
        weights = {}
        for layer in range(num_layers):
            for s, site in enumerate(("attn", "ffn")):
                if ckpt:
                    weights[(layer, site)] = load_checkpoint_layer(
                        ckpt, layer, site, DEVICE
                    )
                else:
                    weights[(layer, site)] = make_weights(
                        HIDDEN, DEVICE, seed=10_000 + 2 * layer + s
                    )
        _SITE_WEIGHTS[key] = (num_layers, weights)
    return _SITE_WEIGHTS[key]


def _run_chain(fused_op, x0, weights, sites, stand_in):
    """One decode step of mHC in the model's order (DeepseekV4Model.forward with
    cross-layer fusion): mhc_pre on the embedding streams, the fused op at every
    site, hc_post after the last layer. Sublayer i's output is
    bf16(layer_input * stand_in[i]). Returns the final [T, 4, D] streams and the
    inputs of every fused call."""
    residual = x0.unsqueeze(1).repeat(1, HC_MULT, 1)
    fn, hc_scale, hc_base, norm_weight = weights[(0, "attn")]
    post, comb, y = mhc_pre(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=RMS_EPS,
        hc_pre_eps=HC_PRE_EPS,
        hc_sinkhorn_eps=HC_SINKHORN_EPS,
        hc_post_mult_value=HC_POST_MULT_VALUE,
        sinkhorn_repeat=SINKHORN_REPEAT,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
    )
    calls = []
    for i, site in enumerate(sites):
        out = (y.float() * stand_in[i]).to(torch.bfloat16)
        # The layer 0 FFN call gets mhc_pre's [T, 4] post, as deepseek_v4.py passes it.
        inputs = (
            out,
            residual,
            post.unsqueeze(-1) if post.ndim == 2 else post,
            comb,
            *weights[site],
        )
        calls.append(inputs)
        residual, post, comb, y = _run(fused_op, *inputs)
    out = (y.float() * stand_in[len(sites)]).to(torch.bfloat16)
    return hc_post(out, residual, post, comb), calls


def _run_chain_ref(x0, weights, sites, stand_in):
    """fp64 version of _run_chain with unrounded streams and sublayer outputs."""
    kw = dict(
        rms_eps=RMS_EPS,
        hc_pre_eps=HC_PRE_EPS,
        hc_sinkhorn_eps=HC_SINKHORN_EPS,
        hc_post_mult_value=HC_POST_MULT_VALUE,
        sinkhorn_repeat=SINKHORN_REPEAT,
        norm_eps=NORM_EPS,
    )

    def cpu(w):
        fn, hc_scale, hc_base, norm_weight = w
        return dict(
            fn=fn.cpu(),
            hc_scale=hc_scale.cpu(),
            hc_base=hc_base.cpu(),
            norm_weight=norm_weight.cpu(),
        )

    residual = x0.cpu().double().unsqueeze(1).repeat(1, HC_MULT, 1)
    post, comb, y = hc_pre_ref_fp64(residual, **cpu(weights[(0, "attn")]), **kw)
    for i, site in enumerate(sites):
        out = y * stand_in[i].cpu().double()
        residual = hc_post_ref_fp64(out, residual, post, comb)
        post, comb, y = hc_pre_ref_fp64(residual, **cpu(weights[site]), **kw)
    out = y * stand_in[len(sites)].cpu().double()
    return hc_post_ref_fp64(out, residual, post, comb)


@pytest.mark.parametrize("t", DECODE_TOKENS)
def test_chain_mhc_pre_to_hc_post(t):
    num_layers, weights = _site_weights()
    sites = fused_call_sites(num_layers)
    g = torch.Generator(device="cpu").manual_seed(500 + t)
    x0 = torch.randn(t, HIDDEN, generator=g).to(DEVICE, torch.bfloat16)
    stand_in = [
        (torch.randn(HIDDEN, generator=g) * 0.5).to(DEVICE)
        for _ in range(len(sites) + 1)
    ]
    label = "checkpoint" if os.environ.get("SGL_MHC_CKPT_DIR") else "random per-site"

    final_new, calls_new = _run_chain(
        mhc_fused_post_pre_small_t, x0, weights, sites, stand_in
    )
    assert final_new.shape == (t, HC_MULT, HIDDEN) and final_new.dtype == torch.bfloat16
    final_ref = _run_chain_ref(x0, weights, sites, stand_in)
    names = ("final_streams",)
    new_stats = error_stats([final_new.cpu()], [final_ref], names=names)

    print(
        f"\nT={t} chain of {len(sites)} fused calls ({label} weights), hc_post output against the fp64 chain"
    )
    envelope = {names[0]: {"max_abs": 0.0, "rel_fro": 0.0}}
    for n_splits in CHAIN_EXISTING_SPLITS:
        existing_op = functools.partial(mhc_fused_post_pre, n_splits=n_splits)
        final_existing, _ = _run_chain(existing_op, x0, weights, sites, stand_in)
        stats = error_stats([final_existing.cpu()], [final_ref], names=names)[names[0]]
        print(
            f"existing n_splits={n_splits:<3} max_abs {stats['max_abs']:.4e} rel_fro {stats['rel_fro']:.4e}"
        )
        for metric in ("max_abs", "rel_fro"):
            envelope[names[0]][metric] = max(envelope[names[0]][metric], stats[metric])
    passed, rows = error_gate(new_stats, envelope, ratio=2.0)
    print("composition check, new against 2x the largest existing error:")
    print(format_gate_rows(rows))
    assert (
        passed
    ), "chain output error exceeds 2x the largest existing-order chain error + 1e-4 against fp64"

    for i, inputs in enumerate(calls_new):
        _gate_cases([inputs], f"T={t} fused call {i}, {sites[i]} ({label} weights)")


@pytest.mark.parametrize("t", DECODE_TOKENS)
def test_error_vs_fp64_recorded(t):
    """The gate on recorded activations (SGL_MHC_ACTS), with the weights the
    recorder saved, else the checkpoint's (SGL_MHC_CKPT_DIR), else random."""
    path = os.environ.get("SGL_MHC_ACTS")
    if not path:
        pytest.skip("SGL_MHC_ACTS is not set")
    ckpt = os.environ.get("SGL_MHC_CKPT_DIR")
    expected_scalars = [
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
    ]
    weights_cache = {}
    sources = set()
    cases = []
    for rec in load_recorded_inputs(path, DEVICE, num_tokens=t):
        if "scalars" in rec:
            assert (
                list(rec["scalars"]) == expected_scalars and rec["norm_eps"] == NORM_EPS
            ), (
                f"recorded call used {rec['scalars']}, norm_eps {rec['norm_eps']}; "
                f"this test runs {expected_scalars}, norm_eps {NORM_EPS}"
            )
        if "fn" in rec:
            weights = (rec["fn"], rec["hc_scale"], rec["hc_base"], rec["norm_weight"])
            sources.add("recorded")
        elif ckpt and "layer" in rec and "site" in rec:
            key = (int(rec["layer"]), rec["site"])
            if key not in weights_cache:
                weights_cache[key] = load_checkpoint_layer(ckpt, key[0], key[1], DEVICE)
            weights = weights_cache[key]
            sources.add("checkpoint")
        else:
            weights = make_weights(rec["x"].shape[1], DEVICE, seed=2000)
            sources.add("random")
        cases.append((rec["x"], rec["residual"], rec["post"], rec["comb"], *weights))
    if not cases:
        pytest.skip(f"no recorded activations with T={t} in {path}")
    _gate_cases(
        cases, f"T={t} recorded activations from {path}, weights: {sorted(sources)}"
    )


@pytest.mark.parametrize("t", DECODE_TOKENS)
def test_deterministic(t):
    inputs = make_inputs(t, HIDDEN, DEVICE, seed=7)
    first = _run(mhc_fused_post_pre_small_t, *inputs)
    for _ in range(3):
        again = _run(mhc_fused_post_pre_small_t, *inputs)
        for a, b in zip(first, again):
            assert torch.equal(a, b)


@pytest.mark.parametrize("t", DECODE_TOKENS)
def test_graph_capture_replay(t):
    if not hasattr(torch.xpu, "XPUGraph") or not hasattr(torch.xpu, "graph"):
        pytest.skip("XPU graph API is unavailable")
    static = make_inputs(t, HIDDEN, DEVICE, seed=11)
    _run(mhc_fused_post_pre_small_t, *static)
    torch.xpu.synchronize()

    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured = _run(mhc_fused_post_pre_small_t, *static)

    for seed in (12, 13):
        fresh = make_inputs(t, HIDDEN, DEVICE, seed=seed)
        for dst, src in zip(static, fresh):
            dst.copy_(src)
        graph.replay()
        torch.xpu.synchronize()
        eager = _run(mhc_fused_post_pre_small_t, *static)
        for a, b in zip(captured, eager):
            assert torch.equal(a, b)


@pytest.mark.parametrize("t", DECODE_TOKENS)
def test_post_and_comb_ranks(t):
    x, residual, post, comb, fn, hc_scale, hc_base, norm_weight = make_inputs(
        t, HIDDEN, DEVICE, seed=3
    )
    base = _run(
        mhc_fused_post_pre_small_t,
        x,
        residual,
        post,
        comb,
        fn,
        hc_scale,
        hc_base,
        norm_weight,
    )
    alt = _run(
        mhc_fused_post_pre_small_t,
        x,
        residual,
        post.squeeze(-1).contiguous(),
        comb.reshape(t, HC_MULT * HC_MULT).contiguous(),
        fn,
        hc_scale,
        hc_base,
        norm_weight,
    )
    for a, b in zip(base, alt):
        assert torch.equal(a, b)


@pytest.mark.parametrize("t", [0, 17, 64])
def test_other_shapes_run_existing_op(t):
    inputs = make_inputs(t, HIDDEN, DEVICE, seed=5)
    new = _run(mhc_fused_post_pre_small_t, *inputs)
    existing = _run(mhc_fused_post_pre, *inputs)
    _check_shapes(new, t, HIDDEN)
    for a, b in zip(new, existing):
        assert torch.equal(a, b)


def test_misaligned_fn_runs_existing_op():
    t = 2
    x, residual, post, comb, fn, hc_scale, hc_base, norm_weight = make_inputs(
        t, HIDDEN, DEVICE, seed=9
    )
    storage = torch.empty(fn.numel() + 1, dtype=fn.dtype, device=DEVICE)
    fn_view = storage[1:].view_as(fn)
    fn_view.copy_(fn)
    assert fn_view.is_contiguous() and fn_view.data_ptr() % 16 != 0
    inputs = (x, residual, post, comb, fn_view, hc_scale, hc_base, norm_weight)
    new = _run(mhc_fused_post_pre_small_t, *inputs)
    existing = _run(mhc_fused_post_pre, *inputs)
    for a, b in zip(new, existing):
        assert torch.equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
