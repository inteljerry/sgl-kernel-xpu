"""Graph-mode micro-benchmark: mhc_fused_post_pre_small_t against the existing op.

Each provider runs a chain of --layers mHC boundaries (85 per DeepSeek-V4-Flash
decode step) where boundary l reads the residual, post and comb written by
boundary l - 1, as in the model. Every boundary has its own fp32 fn [24, 4 D],
so fn streams from DRAM (85 x 1.57 MB at D = 4096) unless --warm reuses one fn.
The chain is captured in one XPU graph and replayed --replays times between two
events. Eager host timing of this op measures the host issue rate, not the
device, so it is not reported.

Output: us per call and ms per step (us per call x --layers / 1000) for every
provider and token count, printed as a table and optionally written to --json.
--launch-baseline also times a graph of 2 x --layers empty kernels (the
in-graph launch cost that every provider pays for its two kernels per call).
--profile prints per-kernel device times of one eager chain from torch.profiler.

Example (one GPU):
    python bench_mhc_fused_post_pre_small_t.py --tokens 1,2,4,8 --json out.json
"""

import argparse
import functools
import json

import torch
from sgl_kernel import mhc_fused_post_pre, mhc_fused_post_pre_small_t
from torch.profiler import ProfilerActivity, profile

HC = 4
HC3 = (2 + HC) * HC
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 20
NORM_EPS = 1e-6

PROVIDERS = {
    "existing": lambda *args, **kwargs: mhc_fused_post_pre(*args, **kwargs),
    "existing_n32": lambda *args, **kwargs: mhc_fused_post_pre(
        *args, n_splits=32, **kwargs
    ),
    "small_t": lambda *args, **kwargs: mhc_fused_post_pre_small_t(*args, **kwargs),
}


def make_chain(t, d, layers, warm, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(t, d, generator=g).to(device=device, dtype=torch.bfloat16)
    residual = (torch.randn(t, HC, d, generator=g) * 4).to(
        device=device, dtype=torch.bfloat16
    )
    post = (2.0 * torch.sigmoid(torch.randn(t, HC, 1, generator=g))).to(device)
    comb = torch.softmax(torch.randn(t, HC, HC, generator=g), dim=-1)
    for _ in range(20):
        comb = comb / comb.sum(dim=-2, keepdim=True)
        comb = comb / comb.sum(dim=-1, keepdim=True)
    comb = comb.to(device)

    weights = []
    for layer in range(1 if warm else layers):
        fn = (torch.randn(HC3, HC * d, generator=g) / (HC * d) ** 0.5).to(device)
        hc_scale = (torch.rand(3, generator=g) * 0.5 + 0.5).to(device)
        hc_base = (torch.randn(HC3, generator=g) * 0.1).to(device)
        norm_weight = (torch.randn(d, generator=g) * 0.2 + 1.0).to(
            device=device, dtype=torch.bfloat16
        )
        weights.append((fn, hc_scale, hc_base, norm_weight))
    if warm:
        weights = weights * layers
    return x, (residual, post, comb), weights


def run_chain(op, x, state, weights):
    residual, post, comb = state
    layer_input = None
    for fn, hc_scale, hc_base, norm_weight in weights:
        residual, post, comb, layer_input = op(
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
            norm_eps=NORM_EPS,
        )
    return residual, post, comb, layer_input


def time_graph(fn_body, replays):
    fn_body()
    fn_body()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        fn_body()
    for _ in range(5):
        graph.replay()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end)
    del graph
    return ms / replays


def launch_baseline(n_kernels, replays, device):
    z = torch.zeros(1, device=device)

    def body():
        for _ in range(n_kernels):
            z.add_(1.0)

    return time_graph(body, replays) * 1000.0 / n_kernels


def profile_chain(op, x, state, weights):
    run_chain(op, x, state, weights)
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        run_chain(op, x, state, weights)
        torch.xpu.synchronize()
    print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=12))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tokens", default="1,2,4,8")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--layers", type=int, default=85, help="mHC boundaries per decode step"
    )
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--providers", default=",".join(PROVIDERS))
    parser.add_argument(
        "--warm", action="store_true", help="reuse one fn so it stays in L2"
    )
    parser.add_argument("--launch-baseline", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    torch.xpu.set_device(torch.device(args.device))
    tokens = [int(v) for v in args.tokens.split(",")]
    providers = args.providers.split(",")
    results = []

    if args.launch_baseline:
        us = launch_baseline(2 * args.layers, args.replays, args.device)
        print(f"empty-kernel launch in graph: {us:.3f} us per kernel")
        results.append({"provider": "empty_kernel", "us_per_kernel": us})

    for t in tokens:
        x, state, weights = make_chain(
            t, args.hidden_size, args.layers, args.warm, args.device
        )
        for name in providers:
            op = PROVIDERS[name]
            ms = time_graph(
                functools.partial(run_chain, op, x, state, weights), args.replays
            )
            us_per_call = ms * 1000.0 / args.layers
            results.append(
                {
                    "provider": name,
                    "T": t,
                    "hidden_size": args.hidden_size,
                    "layers": args.layers,
                    "replays": args.replays,
                    "warm": args.warm,
                    "us_per_call": us_per_call,
                    "ms_per_step": ms,
                }
            )
            if args.profile:
                print(f"--- profile: {name} T={t}")
                profile_chain(op, x, state, weights)
        del x, state, weights
        torch.xpu.empty_cache()

    print(f"\n{'T':>3} {'provider':<12}{'us/call':>10}{'ms/step':>10}")
    for r in results:
        if "T" in r:
            print(
                f"{r['T']:>3} {r['provider']:<12}{r['us_per_call']:>10.2f}{r['ms_per_step']:>10.3f}"
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
