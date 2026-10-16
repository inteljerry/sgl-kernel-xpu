"""CPU check of the small-T mhc_fused_post_pre algorithm (no XPU needed).

fp32 torch emulations of the reduction orders of the two kernel designs are
compared against the fp64 reference:

- small-T: per 64-wide hidden slice, 16 lanes x 4 elements, fn applied to all
  tokens, per-slice partials of the 24 mixes and the 10 Gram entries, then a
  lane-strided second-stage sum; the RMSNorm uses ||y||^2 = pre^T G pre unless
  it is below 1/64 of (sum_k pre_k ||r'_k||)^2, where ||y||^2 is summed directly.
- existing: split-K FMA partials (8 splits for T < 8, 4 at T = 8) summed over
  sub-groups and splits, and a direct sum of y^2 by 80 work-items.

Both emulations use the same precise fp32 math for sigmoid, rsqrt and the
Sinkhorn, so they differ only in summation order and in how ||y||^2 is formed.
This checks the math of the design; the SYCL kernel itself is checked by
test_mhc_fused_post_pre_small_t.py on an XPU.
"""

import pytest
import torch
from mhc_reference import (
    HC_MULT,
    cancelling_inputs,
    error_gate,
    error_stats,
    make_inputs,
    mhc_fused_post_pre_ref_fp64,
)

RMS_EPS = 1e-6
HC_EPS = 1e-6
POST_MULT = 2.0
NORM_EPS = 1e-6
GRAM_CANCEL_LIMIT = 1.0 / 64.0  # kGramCancelLimit in MHCFusedPostPreSmallT.cpp


_GRAM_DIAG = (0, 4, 7, 9)  # packed upper-triangle index of G_jj


def _lane_tree_sum(v):
    """Result of the xor 8, 4, 2, 1 butterfly over the last (16-lane) dim."""
    v = v[..., :8] + v[..., 8:]
    v = v[..., :4] + v[..., 4:]
    v = v[..., :2] + v[..., 2:]
    return v[..., 0] + v[..., 1]


def _post_bf16(x, residual, post, comb):
    t = x.shape[0]
    p = post.reshape(t, HC_MULT).float()
    c = comb.reshape(t, HC_MULT, HC_MULT).float()
    out = p[:, :, None] * x.float()[:, None, :]
    for k in range(HC_MULT):
        out = out + c[:, k, :, None] * residual.float()[:, k, None, :]
    return out.to(torch.bfloat16)


def _finish(tot_mix, sqrsum, y_sqr_fn, streams, hc_scale, hc_base, norm_weight, d):
    """Shared fp32 tail: pre, post, Sinkhorn comb and layer_input."""
    rms = torch.rsqrt(sqrsum / (HC_MULT * d) + RMS_EPS)[:, None]
    mixes = tot_mix * rms
    s, b = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(mixes[:, :4] * s[0] + b[:4]) + HC_EPS
    post = POST_MULT * torch.sigmoid(mixes[:, 4:8] * s[1] + b[4:8])
    comb = (mixes[:, 8:] * s[2] + b[8:]).view(-1, 4, 4)
    comb = torch.softmax(comb, dim=-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)

    y = torch.zeros_like(streams[:, 0])
    for k in range(HC_MULT):
        y = y + pre[:, k, None] * streams[:, k]
    if norm_weight is None:
        return post.unsqueeze(-1), comb, y.to(torch.bfloat16)
    y_scale = torch.rsqrt(y_sqr_fn(pre, y) / d + NORM_EPS)[:, None]
    return (
        post.unsqueeze(-1),
        comb,
        (y * y_scale * norm_weight.float()).to(torch.bfloat16),
    )


def small_t_emulation(
    x,
    residual,
    post,
    comb,
    fn,
    hc_scale,
    hc_base,
    norm_weight,
    gram_limit=GRAM_CANCEL_LIMIT,
    info=None,
):
    """gram_limit 0 disables the direct ||y||^2 fallback; info, if a dict, gets
    the per-token flag "direct" (fallback taken)."""
    t, _, d = residual.shape
    n_slices = d // 64
    residual_cur = _post_bf16(x, residual, post, comb)
    streams = residual_cur.float()

    rs = streams.view(t, HC_MULT, n_slices, 16, 4)
    fnv = fn.float().view(24, HC_MULT, n_slices, 16, 4)
    acc = torch.zeros(t, 24, n_slices, 16)
    for k in range(HC_MULT):
        for i in range(4):
            acc = acc + fnv[None, :, k, :, :, i] * rs[:, None, k, :, :, i]
    parts = [_lane_tree_sum(acc)]
    for j in range(HC_MULT):
        for k in range(j, HC_MULT):
            g = torch.zeros(t, n_slices, 16)
            for i in range(4):
                g = g + rs[:, j, :, :, i] * rs[:, k, :, :, i]
            parts.append(_lane_tree_sum(g)[:, None])
    part = torch.cat(parts, dim=1)  # [T, 34, n_slices]

    lanes = torch.zeros(t, part.shape[1], 16)
    for c in range(0, n_slices, 16):
        lanes = lanes + part[..., c : c + 16]
    tot = _lane_tree_sum(lanes)
    gram = tot[:, 24:]
    sqrsum = gram[:, 0] + gram[:, 4] + gram[:, 7] + gram[:, 9]

    def y_sqr(pre, y):
        total = torch.zeros(t)
        bound = torch.zeros(t)
        n = 0
        for j in range(HC_MULT):
            bound = bound + pre[:, j] * gram[:, _GRAM_DIAG[j]].sqrt()
            for k in range(j, HC_MULT):
                w = 1.0 if j == k else 2.0
                total = total + w * pre[:, j] * pre[:, k] * gram[:, n]
                n += 1
        direct = total < gram_limit * bound * bound
        if info is not None:
            info["direct"] = direct
        # Direct sum: 64 work-items stride over 4-wide vectors, sub-group
        # butterflies, then the 4 sub-groups in order.
        yv = y.view(t, d // 256, 64, 4)
        local = torch.zeros(t, 64)
        for c in range(yv.shape[1]):
            for i in range(4):
                local = local + yv[:, c, :, i] ** 2
        sg = _lane_tree_sum(local.view(t, 4, 16))
        direct_sum = ((sg[:, 0] + sg[:, 1]) + sg[:, 2]) + sg[:, 3]
        return torch.where(direct, direct_sum, total).clamp_min(0.0)

    tail = _finish(
        tot[:, :24], sqrsum, y_sqr, streams, hc_scale, hc_base, norm_weight, d
    )
    return (residual_cur,) + tail


def existing_emulation(
    x, residual, post, comb, fn, hc_scale, hc_base, norm_weight, n_splits=0
):
    """n_splits 0 is the existing default for T <= 32 (8 for T < 8 at D <= 4096, else 4)."""
    t, _, d = residual.shape
    if n_splits <= 0:
        n_splits = 8 if (t < 8 and d <= 4096) else 4
    residual_cur = _post_bf16(x, residual, post, comb)
    streams = residual_cur.float()

    # FMA kernel: work-items of 128 x 4 elements per split, per-item sums over
    # k then i, sub-group butterfly, then the 8 sub-groups in order, then splits.
    # Each split is zero-padded to whole 512-element rounds (adding 0 is exact).
    assert d % (4 * n_splits) == 0
    per_split = d // n_splits
    iters = -(-per_split // 512)
    pad = iters * 512 - per_split

    def split_view(v, lead):
        v = v.reshape(*lead, n_splits, per_split)
        v = torch.nn.functional.pad(v, (0, pad))
        return v.view(*lead, n_splits, iters, 8, 16, 4)

    rs = split_view(streams, (t, HC_MULT))
    fnv = split_view(fn.float(), (24, HC_MULT))
    mix = torch.zeros(t, 24, n_splits, 8, 16)
    sq = torch.zeros(t, n_splits, 8, 16)
    for it in range(iters):
        a = torch.zeros(t, 24, n_splits, 8, 16)
        for k in range(HC_MULT):
            for i in range(4):
                a = a + fnv[None, :, k, :, it, :, :, i] * rs[:, None, k, :, it, :, :, i]
        mix = mix + a
        for j in range(HC_MULT):
            for i in range(4):
                sq = sq + rs[:, j, :, it, :, :, i] ** 2
    mix_sg, sq_sg = _lane_tree_sum(mix), _lane_tree_sum(sq)
    mix_split = torch.zeros(t, 24, n_splits)
    sq_split = torch.zeros(t, n_splits)
    for s in range(8):
        mix_split = mix_split + mix_sg[..., s]
        sq_split = sq_split + sq_sg[..., s]
    tot_mix = torch.zeros(t, 24)
    sqrsum = torch.zeros(t)
    for s in range(n_splits):
        tot_mix = tot_mix + mix_split[..., s]
        sqrsum = sqrsum + sq_split[..., s]

    def y_sqr(_pre, y):
        # 80 work-items stride over 8-wide vectors, then 5 sub-group sums.
        nvec = d // 8
        yv = y.view(t, nvec, 8)
        local = torch.zeros(t, 80)
        for c in range(0, nvec, 80):
            chunk = yv[:, c : c + 80]
            for j in range(8):
                local[:, : chunk.shape[1]] = (
                    local[:, : chunk.shape[1]] + chunk[:, :, j] ** 2
                )
        sg_sums = _lane_tree_sum(local.view(t, 5, 16))
        lanes = torch.zeros(t, 16)
        lanes[:, :5] = sg_sums
        return _lane_tree_sum(lanes)

    tail = _finish(tot_mix, sqrsum, y_sqr, streams, hc_scale, hc_base, norm_weight, d)
    return (residual_cur,) + tail


def _ref(inputs):
    x, residual, post, comb, fn, hc_scale, hc_base, norm_weight = inputs
    return mhc_fused_post_pre_ref_fp64(
        x,
        residual,
        post,
        comb,
        fn,
        hc_scale,
        hc_base,
        rms_eps=RMS_EPS,
        hc_pre_eps=HC_EPS,
        hc_sinkhorn_eps=HC_EPS,
        hc_post_mult_value=POST_MULT,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
    )


def test_gram_identity_fp64():
    """||sum_k pre_k r_k||^2 == pre^T G pre with G_jk = <r_j, r_k>."""
    g = torch.Generator().manual_seed(0)
    r = torch.randn(3, HC_MULT, 4096, generator=g, dtype=torch.float64) * 4
    pre = torch.rand(3, HC_MULT, generator=g, dtype=torch.float64)
    y = (pre[:, :, None] * r).sum(1)
    gram = torch.einsum("tjd,tkd->tjk", r, r)
    torch.testing.assert_close(
        y.square().sum(-1),
        torch.einsum("tj,tjk,tk->t", pre, gram, pre),
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize("t", [1, 2, 4, 8])
@pytest.mark.parametrize("d", [4096, 7168])
def test_small_t_emulation_close_to_fp64(t, d):
    inputs = make_inputs(t, d, "cpu", seed=t)
    out = small_t_emulation(*inputs)
    ref = _ref(inputs)
    # bf16 outputs: within 2 bf16 ulps of the reference magnitude; fp32 outputs: 1e-4.
    for name, o, r, tol in zip(
        ("residual_cur", "post", "comb", "layer_input"),
        out,
        ref,
        (1e-2, 1e-4, 1e-4, 1e-2),
    ):
        torch.testing.assert_close(
            o.double(), r, rtol=tol, atol=tol, msg=f"{name} (T={t}, D={d})"
        )


@pytest.mark.parametrize("t", [1, 2, 4, 8])
def test_small_t_emulation_error_gate(t):
    """Preview of the error gate on the emulated reduction orders."""
    new_out, existing_out, refs = (
        [[] for _ in range(4)],
        [[] for _ in range(4)],
        [[] for _ in range(4)],
    )
    for seed in range(4):
        inputs = make_inputs(t, 4096, "cpu", seed=100 + seed)
        ref = _ref(inputs)
        for acc, out in (
            (new_out, small_t_emulation(*inputs)),
            (existing_out, existing_emulation(*inputs)),
        ):
            for i in range(4):
                acc[i].append(out[i])
        for i in range(4):
            refs[i].append(ref[i])
    passed, rows = error_gate(
        error_stats(new_out, refs), error_stats(existing_out, refs)
    )
    print(f"\nT={t} (emulated reduction orders, fp32 on CPU)")
    for r in rows:
        print(
            f"  {r['output']:<14}{r['metric']:<9}{r['new']:>12.4e}{r['existing']:>12.4e}{r['ratio']:>8.3f}  {r['ok']}"
        )
    assert passed


@pytest.mark.parametrize("cancel", [1e-1, 1e-2, 1e-3])
def test_gram_cancellation_falls_back_to_direct_sum(cancel):
    """With cancelling streams the small-T RMSNorm takes the direct ||y||^2 sum and
    stays within the gate of the existing order; the pure Gram path is printed for
    comparison."""
    t, d = 4, 4096
    new_out, gram_out, existing_out, refs = (
        [[] for _ in range(4)],
        [[] for _ in range(4)],
        [[] for _ in range(4)],
        [[] for _ in range(4)],
    )
    for seed in range(3):
        inputs = cancelling_inputs(t, d, cancel, seed=300 + seed)
        info = {}
        outs = (
            (new_out, small_t_emulation(*inputs, info=info)),
            (gram_out, small_t_emulation(*inputs, gram_limit=0.0)),
            (existing_out, existing_emulation(*inputs)),
        )
        assert bool(info["direct"].all()), "cancelling streams must take the direct sum"
        ref = _ref(inputs)
        for acc, out in outs:
            for i in range(4):
                acc[i].append(out[i])
        for i in range(4):
            refs[i].append(ref[i])
    existing_stats = error_stats(existing_out, refs)
    passed, rows = error_gate(error_stats(new_out, refs), existing_stats)
    _, gram_rows = error_gate(error_stats(gram_out, refs), existing_stats)
    print(f"\ncancel={cancel}: small-T with the direct fallback")
    for r in rows[-2:]:
        print(
            f"  {r['output']:<14}{r['metric']:<9}{r['new']:>12.4e}{r['existing']:>12.4e}{r['ratio']:>8.3f}  {r['ok']}"
        )
    print("  Gram only (no fallback):")
    for r in gram_rows[-2:]:
        print(
            f"  {r['output']:<14}{r['metric']:<9}{r['new']:>12.4e}{r['existing']:>12.4e}{r['ratio']:>8.3f}  {r['ok']}"
        )
    assert passed


@pytest.mark.parametrize("t", [1, 8])
def test_random_streams_keep_gram_path(t):
    """Uncorrelated streams are far from the cancellation limit (ratio about 1/4)."""
    info = {}
    small_t_emulation(*make_inputs(t, 4096, "cpu", seed=40 + t), info=info)
    assert not bool(info["direct"].any())
