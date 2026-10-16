"""fp64 reference and error metrics for mhc_fused_post_pre (HC = 4).

The op is hc_post of one sublayer followed by hc_pre of the next:

    r'_j = post_j * x + sum_i comb[i, j] * r_i                (stored as bf16)
    mixes = (flatten(r') @ fn.T) * rsqrt(mean(flatten(r')^2) + rms_eps)
    pre = sigmoid(mixes[0:4] * scale[0] + base[0:4]) + hc_pre_eps
    post = hc_post_mult_value * sigmoid(mixes[4:8] * scale[1] + base[4:8])
    comb = sinkhorn(softmax_rows(mixes[8:24] * scale[2] + base[8:24]))
    y = sum_k pre_k * r'_k, then y * rsqrt(mean(y^2) + norm_eps) * norm_weight

hc_pre reads the bf16 streams the op returns, so the reference rounds r' to
bf16 before the pre part (round_residual=True). The residual_cur output is
compared against the unrounded fp64 r'. A kernel whose fp32 r' lands on the
other side of a bf16 rounding midpoint than the fp64 r' reads a different
stream element; passing that kernel's own residual_cur as `streams` gives a
reference for the pre part alone.

hc_post_ref_fp64 and hc_pre_ref_fp64 are the two halves; hc_pre_ref_fp64 is
also the reference of mhc_pre (the layer 0 attention boundary, where no hc_post
precedes the pre part).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch

HC_MULT = 4
HC_MULT3 = (2 + HC_MULT) * HC_MULT  # 24
# DeepSeek-V4-Flash: 43 decoder layers, hidden 4096, fn [24, 16384].
NUM_LAYERS = 43


def hc_post_ref_fp64(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """fp64 r'_j = post_j * x + sum_i comb[i, j] * r_i, [T, 4, D]."""
    t, hc, _ = residual.shape
    assert hc == HC_MULT
    post64 = post_layer_mix.reshape(t, hc).double()
    comb64 = comb_res_mix.reshape(t, hc, hc).double()
    return post64[:, :, None] * x.double()[:, None, :] + torch.einsum(
        "tij,tid->tjd", comb64, residual.double()
    )


def hc_pre_ref_fp64(
    streams: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp64 hc_pre of streams [T, 4, D]: (post [T, 4], comb [T, 4, 4], layer_input [T, D])."""
    t, hc, d = streams.shape
    assert hc == HC_MULT
    streams = streams.double()
    flat = streams.reshape(t, hc * d)
    rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = (flat @ fn.double().T) * rms

    scale = hc_scale.double().flatten()
    base = hc_base.double().flatten()
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_pre_eps
    post = hc_post_mult_value * torch.sigmoid(
        mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    )
    comb = (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).view(t, hc, hc)
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    y = (pre[:, :, None] * streams).sum(dim=1)
    if norm_weight is not None:
        eps = 1e-6 if norm_eps is None else norm_eps
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
        y = y * norm_weight.double()
    return post, comb, y


def mhc_fused_post_pre_ref_fp64(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: Optional[float] = None,
    round_residual: bool = True,
    streams: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns fp64 (residual_cur [T, 4, D], post [T, 4, 1], comb [T, 4, 4], layer_input [T, D]).

    streams, if given, are the bf16 streams the pre part reads (for example a
    kernel's own residual_cur output), instead of the rounded fp64 r'."""
    residual_cur = hc_post_ref_fp64(x, residual, post_layer_mix, comb_res_mix)
    if streams is None:
        streams = residual_cur.to(torch.bfloat16) if round_residual else residual_cur
    post, comb, y = hc_pre_ref_fp64(
        streams,
        fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return residual_cur, post.unsqueeze(-1), comb, y


OUTPUT_NAMES = ("residual_cur", "post_mix", "comb_mix", "layer_input")


def error_stats(outputs, refs, names=OUTPUT_NAMES) -> Dict[str, Dict[str, float]]:
    """Error of each output against its fp64 reference, relative to the reference RMS.

    max_abs is max |out - ref| / rms(ref) and rel_fro is ||out - ref|| / ||ref||
    (= rms(out - ref) / rms(ref)). outputs and refs are sequences of tensors (one
    per name), or of lists of tensors (pooled over seeds or layers by
    concatenation)."""
    stats = {}
    for name, out, ref in zip(names, outputs, refs):
        if isinstance(out, (list, tuple)):
            out = torch.cat([o.reshape(-1) for o in out])
            ref = torch.cat([r.reshape(-1) for r in ref])
        ref = ref.detach().double().cpu().reshape(-1)
        diff = out.detach().double().cpu().reshape(-1) - ref
        ref_rms = ref.square().mean().sqrt().item() if ref.numel() else 0.0
        scale = ref_rms if ref_rms > 0 else 1.0
        stats[name] = {
            "max_abs": diff.abs().max().item() / scale if diff.numel() else 0.0,
            "rel_fro": (
                diff.square().mean().sqrt().item() / scale if diff.numel() else 0.0
            ),
        }
    return stats


def error_gate(new_stats, existing_stats, ratio: float = 1.25, abs_tol: float = 1e-4):
    """The error gate: reject if, on any output, the new kernel's error
    exceeds `ratio` x the existing kernel's error + `abs_tol`, by max-abs or by
    Frobenius norm (both relative to the reference RMS, see error_stats).
    Returns (passed, rows) with one row per output and metric."""
    rows = []
    passed = True
    for name in new_stats:
        for metric in ("max_abs", "rel_fro"):
            new = new_stats[name][metric]
            existing = existing_stats[name][metric]
            ok = new <= ratio * existing + abs_tol
            passed = passed and ok
            rows.append(
                {
                    "output": name,
                    "metric": metric,
                    "new": new,
                    "existing": existing,
                    "ratio": (
                        new / existing
                        if existing > 0
                        else float("inf") if new > 0 else 1.0
                    ),
                    "ok": ok,
                }
            )
    return passed, rows


def format_gate_rows(rows) -> str:
    lines = [f"{'output':<14}{'metric':<9}{'new':>12}{'existing':>12}{'ratio':>8}  ok"]
    for r in rows:
        lines.append(
            f"{r['output']:<14}{r['metric']:<9}{r['new']:>12.4e}{r['existing']:>12.4e}{r['ratio']:>8.3f}  {r['ok']}"
        )
    return "\n".join(lines)


def cancelling_inputs(t: int, d: int, cancel: float, seed: int):
    """Inputs whose new streams cancel under the pre weights: post = 0 and comb = I
    give r' = r, fn rows 0..3 = 0 give equal pre_k, and r_1 = -r_0 + cancel * n,
    r_3 = -r_2 + cancel * n', so ||y|| / sum_k pre_k ||r_k|| is about cancel / 4.
    Returns CPU tensors."""
    x, residual, post, comb, fn, hc_scale, hc_base, norm_weight = make_inputs(
        t, d, "cpu", seed=seed
    )
    g = torch.Generator().manual_seed(seed + 1)
    r0 = torch.randn(t, d, generator=g) * 4
    r2 = torch.randn(t, d, generator=g) * 4
    r1 = -r0 + cancel * torch.randn(t, d, generator=g)
    r3 = -r2 + cancel * torch.randn(t, d, generator=g)
    residual = torch.stack([r0, r1, r2, r3], dim=1).to(torch.bfloat16)
    post = torch.zeros_like(post)
    comb = torch.eye(HC_MULT).expand(t, HC_MULT, HC_MULT).contiguous()
    fn = fn.clone()
    fn[:HC_MULT] = 0.0
    hc_base = hc_base.clone()
    hc_base[:HC_MULT] = 0.3
    return x, residual, post, comb, fn, hc_scale, hc_base, norm_weight


def make_weights(d: int, device, seed: int = 0, with_norm: bool = True):
    """Random mHC weights of one pre-mix site at production shapes: fp32 fn
    [24, 4 D] with the 1 / sqrt(4 D) scale of a trained projection, hc_scale [3],
    hc_base [24] and a bf16 norm weight [D]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    hc_hidden = HC_MULT * d
    fn = torch.randn(HC_MULT3, hc_hidden, generator=g) / (hc_hidden**0.5)
    hc_scale = torch.rand(3, generator=g) * 0.5 + 0.5
    hc_base = torch.randn(HC_MULT3, generator=g) * 0.1
    norm_weight = (
        (torch.randn(d, generator=g) * 0.2 + 1.0).to(torch.bfloat16)
        if with_norm
        else None
    )
    return tuple(
        None if v is None else v.to(device).contiguous()
        for v in (fn, hc_scale, hc_base, norm_weight)
    )


def make_inputs(
    t: int,
    d: int,
    device,
    seed: int = 0,
    residual_scale: float = 4.0,
    with_norm: bool = True,
):
    """Scaled random inputs at production shapes: bf16 x [T, D] and residual [T, 4, D],
    post in (0, 2) like 2 * sigmoid, a doubly stochastic comb, fp32 fn [24, 4 D]
    with the 1 / sqrt(4 D) scale of a trained projection, and a bf16 norm weight."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    hc_hidden = HC_MULT * d

    x = torch.randn(t, d, generator=g).to(torch.bfloat16)
    residual = (torch.randn(t, HC_MULT, d, generator=g) * residual_scale).to(
        torch.bfloat16
    )
    post = 2.0 * torch.sigmoid(torch.randn(t, HC_MULT, 1, generator=g))
    comb = torch.softmax(torch.randn(t, HC_MULT, HC_MULT, generator=g), dim=-1)
    for _ in range(20):
        comb = comb / comb.sum(dim=-2, keepdim=True)
        comb = comb / comb.sum(dim=-1, keepdim=True)
    fn = torch.randn(HC_MULT3, hc_hidden, generator=g) / (hc_hidden**0.5)
    hc_scale = torch.rand(3, generator=g) * 0.5 + 0.5
    hc_base = torch.randn(HC_MULT3, generator=g) * 0.1
    norm_weight = (
        (torch.randn(d, generator=g) * 0.2 + 1.0).to(torch.bfloat16)
        if with_norm
        else None
    )

    def dev(v):
        return None if v is None else v.to(device).contiguous()

    return tuple(
        dev(v)
        for v in (
            x,
            residual,
            post,
            comb.float(),
            fn.float(),
            hc_scale.float(),
            hc_base.float(),
            norm_weight,
        )
    )


def fused_call_sites(num_layers: int = NUM_LAYERS) -> List[Tuple[int, str]]:
    """(layer, site) of every mhc_fused_post_pre call of one decode step, in order.

    The attention pre-mix of layer 0 runs mhc_pre on the embedding streams (no
    hc_post precedes it), so the first fused call is the FFN pre-mix of layer 0
    (hc_post of the layer 0 attention). Then each layer l >= 1 has its attention
    pre-mix (hc_post of the layer l - 1 FFN) and its FFN pre-mix. The streams of
    the last call are closed by hc_post after the last layer. 43 layers: 85 calls."""
    sites = [(0, "ffn")]
    for layer in range(1, num_layers):
        sites += [(layer, "attn"), (layer, "ffn")]
    return sites


def checkpoint_num_layers(ckpt_dir: str) -> int:
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        return int(json.load(f)["num_hidden_layers"])


def load_recorded_inputs(path: str, device, num_tokens: Optional[int] = None):
    """Activations recorded at mhc_fused_post_pre call sites of a running model.

    path is either
    - a directory written by SGLang's SGLANG_XPU_MHC_RECORD_DIR recorder: files
      mhc-f<forward>-t<tokens>.pt, each a list of the records of one forward's
      fused calls, and weights/layer<layer>-<site>.pt with that site's fn,
      hc_scale, hc_base and norm_weight as the server held them; or
    - one torch.save'd list of records.

    A record has x [T, D] bf16, residual [T, 4, D] bf16, post [T, 4] or
    [T, 4, 1] fp32, comb [T, 4, 4] fp32, and optionally layer (int), site
    ("attn" or "ffn"), scalars ([rms_eps, hc_pre_eps, hc_sinkhorn_eps,
    hc_post_mult_value, sinkhorn_repeat]) and norm_eps. Records from a recorder
    directory also get fn, hc_scale, hc_base and norm_weight. Returns the records
    with T == num_tokens (all if None), tensors on device."""
    weights = {}
    if os.path.isdir(path):
        records = []
        for name in sorted(os.listdir(path)):
            if name.startswith("mhc-f") and name.endswith(".pt"):
                t = int(name[: -len(".pt")].rsplit("-t", 1)[1])
                if num_tokens is None or t == num_tokens:
                    records += torch.load(
                        os.path.join(path, name), map_location="cpu", weights_only=True
                    )
        weights_dir = os.path.join(path, "weights")
        if os.path.isdir(weights_dir):
            for name in os.listdir(weights_dir):
                layer, site = name[len("layer") : -len(".pt")].split("-")
                weights[(int(layer), site)] = torch.load(
                    os.path.join(weights_dir, name),
                    map_location="cpu",
                    weights_only=True,
                )
    else:
        records = torch.load(path, map_location="cpu", weights_only=True)

    def dev(v):
        return None if v is None else v.to(device).contiguous()

    device_weights = {}
    out = []
    for rec in records:
        if num_tokens is not None and rec["x"].shape[0] != num_tokens:
            continue
        item = {k: dev(v) if torch.is_tensor(v) else v for k, v in rec.items()}
        key = (int(rec.get("layer", -1)), rec.get("site"))
        if key in weights:
            if key not in device_weights:
                device_weights[key] = {k: dev(v) for k, v in weights[key].items()}
            item.update(device_weights[key])
        out.append(item)
    return out


def load_checkpoint_layer(ckpt_dir: str, layer: int, site: str, device):
    """Real mHC weights of one layer from a DeepSeek-V4 safetensors checkpoint.

    site "attn" is the attention pre-mix (hc_attn_*, attn_norm), "ffn" the FFN
    pre-mix (hc_ffn_*, ffn_norm). Returns (fn, hc_scale, hc_base, norm_weight)."""
    from safetensors import safe_open

    names = {
        "fn": f"layers.{layer}.hc_{site}_fn",
        "scale": f"layers.{layer}.hc_{site}_scale",
        "base": f"layers.{layer}.hc_{site}_base",
        "norm": f"layers.{layer}.{site}_norm.weight",
    }
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    out = {}
    for key, name in names.items():
        with safe_open(os.path.join(ckpt_dir, weight_map[name]), framework="pt") as f:
            out[key] = f.get_tensor(name)
    return (
        out["fn"].float().to(device).contiguous(),
        out["scale"].float().to(device).contiguous(),
        out["base"].float().to(device).contiguous(),
        out["norm"].to(torch.bfloat16).to(device).contiguous(),
    )
