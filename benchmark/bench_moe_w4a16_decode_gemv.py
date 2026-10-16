# Cold-weight micro-benchmark: decode MXFP4 W4A16 MoE GEMV
# (moe_w4a16_decode_gemv) against the existing path (fused_experts with
# use_mxfp4_w4a16=True), at DeepSeek-V4-Flash routed-expert shapes per GPU at
# TP 8 (E 256, topk 6, H 4096, I 256), decode batches T = 1, 2, 4, 8.
#
# Cold weights: --layers copies of the full expert set (about 428 MB each, like
# the 43 MoE layers of a decode step) and a random routing per call. Call c
# reads layer c % layers with routing c, so between two reads of one layer's
# weights the other layers stream several hundred MB, far above the last-level
# cache. A synthetic loop on one weight set would read cache-resident weights.
#
# Timing:
#   eager  XPU events around a window of --calls calls.
#   graph  one torch.xpu.XPUGraph capturing --calls calls, replayed
#          --replays times; per-call time = replay time / calls. This is how
#          the decode step runs (graph replay of every MoE layer).
# The GPU clock idles low; --warm-seconds of back-to-back calls run first.
#
# Reported per T and provider: us per call (eager, graph), distinct experts
# per call, MB of expert weights and scales read per call, the effective read
# bandwidth in graph mode, and the speedup against the existing path.
#
# Run (one GPU):
#   python benchmark/bench_moe_w4a16_decode_gemv.py
#   python benchmark/bench_moe_w4a16_decode_gemv.py --tokens 1 8 --json-out out.json

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import sgl_kernel  # noqa: F401 - registers torch.ops.sgl_kernel
import torch
from sgl_kernel import fused_experts, moe_w4a16_decode_gemv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from moe_w4a16_decode_reference import make_routing  # noqa: E402

E, H, I, TOPK = 256, 4096, 256, 6
RSF, LIMIT = 1.5, 10.0
DEVICE = "xpu"

W13_BYTES_PER_EXPERT = 2 * I * (H // 2 + H // 32)
W2_BYTES_PER_EXPERT = H * (I // 2 + I // 32)


def _existing(x, layer, tw, ids):
    w13, s13, w2, s2 = layer
    return fused_experts(
        x,
        w13,
        w2,
        tw,
        ids,
        activation="silu",
        use_mxfp4_w4a16=True,
        w1_scale=s13,
        w2_scale=s2,
        routed_scaling_factor=RSF,
        swiglu_limit=LIMIT,
    )


def _gemv(x, layer, tw, ids):
    w13, s13, w2, s2 = layer
    return moe_w4a16_decode_gemv(
        x, w13, w2, tw, ids, s13, s2, routed_scaling_factor=RSF, swiglu_limit=LIMIT
    )


PROVIDERS = {"existing": _existing, "gemv": _gemv}


def _make_layers(n):
    # Content does not change the time of either op; random bytes and scales
    # in the real exponent range are enough.
    layers = []
    for _ in range(n):
        w13 = torch.randint(
            -128, 128, (E, 2 * I, H // 2), dtype=torch.int8, device=DEVICE
        )
        s13 = torch.randint(
            115, 126, (E, 2 * I, H // 32), dtype=torch.uint8, device=DEVICE
        )
        w2 = torch.randint(-128, 128, (E, H, I // 2), dtype=torch.int8, device=DEVICE)
        s2 = torch.randint(115, 126, (E, H, I // 32), dtype=torch.uint8, device=DEVICE)
        layers.append(
            (w13, s13.view(torch.float8_e8m0fnu), w2, s2.view(torch.float8_e8m0fnu))
        )
    return layers


def _make_calls(num_tokens, calls, layers, seed):
    out = []
    for c in range(calls):
        ids, tw = make_routing(num_tokens, E, TOPK, "random", seed + c)
        x = (torch.randn((num_tokens, H)) * 1.0).to(torch.bfloat16)
        out.append(
            {
                "layer": layers[c % len(layers)],
                "x": x.to(DEVICE),
                "ids": ids.to(DEVICE),
                "tw": tw.to(DEVICE),
                "experts": len(set(ids.view(-1).tolist())),
            }
        )
    return out


def _run_calls(fn, calls):
    for c in calls:
        fn(c["x"], c["layer"], c["tw"], c["ids"])


def _eager_us(fn, calls, repetitions):
    samples = []
    for _ in range(repetitions):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        _run_calls(fn, calls)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / len(calls))
    return statistics.median(samples)


def _graph_us(fn, calls, replays):
    _run_calls(fn, calls)  # size the workspaces outside capture
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(xpu_graph=graph):
        _run_calls(fn, calls)
    graph.replay()
    torch.xpu.synchronize()
    samples = []
    for _ in range(replays):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / len(calls))
    del graph
    return statistics.median(samples)


def _warm(fn, calls, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        _run_calls(fn, calls)
        torch.xpu.synchronize()


def _check(calls):
    worst = 0.0
    for c in calls[:4]:
        a = _gemv(c["x"], c["layer"], c["tw"], c["ids"]).float()
        b = _existing(c["x"], c["layer"], c["tw"], c["ids"]).float()
        rel = ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
        if not math.isfinite(rel):
            raise RuntimeError(f"non-finite relative error {rel} between the providers")
        worst = max(worst, rel)
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--calls", type=int, default=43)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--warm-seconds", type=float, default=3.0)
    parser.add_argument(
        "--providers", nargs="+", default=list(PROVIDERS), choices=list(PROVIDERS)
    )
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    layers = _make_layers(args.layers)
    results = []
    for num_tokens in args.tokens:
        calls = _make_calls(num_tokens, args.calls, layers, seed=1000 * num_tokens)
        experts = statistics.mean(c["experts"] for c in calls)
        mb = experts * (W13_BYTES_PER_EXPERT + W2_BYTES_PER_EXPERT) / 1e6
        rel = _check(calls)
        row = {
            "tokens": num_tokens,
            "experts_per_call": experts,
            "expert_mb_per_call": mb,
            "gemv_vs_existing_rel_fro": rel,
        }
        for name in args.providers:
            fn = PROVIDERS[name]
            _warm(fn, calls, args.warm_seconds)
            row[f"{name}_eager_us"] = _eager_us(fn, calls, args.repetitions)
            if not args.no_graph:
                row[f"{name}_graph_us"] = _graph_us(fn, calls, args.replays)
                row[f"{name}_graph_gbps"] = (
                    mb * 1e6 / (row[f"{name}_graph_us"] * 1e-6) / 1e9
                )
        for mode in ("eager", "graph"):
            if f"existing_{mode}_us" in row and f"gemv_{mode}_us" in row:
                row[f"speedup_{mode}"] = (
                    row[f"existing_{mode}_us"] / row[f"gemv_{mode}_us"]
                )
        results.append(row)
        print(json.dumps(row))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "shape": {"E": E, "H": H, "I": I, "topk": TOPK},
                    "args": vars(args),
                    "results": results,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
