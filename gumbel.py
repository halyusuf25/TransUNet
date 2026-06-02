import time
import math
import torch


# =========================================================
# 1) The TWO recommended implementations
# =========================================================
@torch.no_grad()
def sample_gumbel_manual_safe(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """
    Dynamic-shape friendly, fast, and numerically safer for mixed precision:
    - sample U in fp32
    - compute Gumbel in fp32
    - cast back to x.dtype
    """
    u = torch.rand_like(x, dtype=torch.float32)
    u = u.clamp(min=eps, max=1.0 - eps)
    g = -torch.log(-torch.log(u))
    return g.to(dtype=x.dtype)


class GumbelSampler:
    """
    Prebuilt scalar torch.distributions.Gumbel(0,1), dynamic-shape friendly.
    Reuses the distribution object and samples with sample_shape=x.shape.
    """
    def __init__(self, device: torch.device):
        self.dist = torch.distributions.Gumbel(
            torch.tensor(0.0, device=device, dtype=torch.float32),
            torch.tensor(1.0, device=device, dtype=torch.float32),
        )

    @torch.no_grad()
    def sample_like(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist.sample(x.shape).to(dtype=x.dtype)


# =========================================================
# 2) Benchmark utilities
# =========================================================
def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _format_num(n: float) -> str:
    # Human-friendly large number formatter
    if n >= 1e12:
        return f"{n/1e12:.3f}T"
    if n >= 1e9:
        return f"{n/1e9:.3f}G"
    if n >= 1e6:
        return f"{n/1e6:.3f}M"
    if n >= 1e3:
        return f"{n/1e3:.3f}K"
    return f"{n:.3f}"


@torch.no_grad()
def benchmark_sampler(
    fn,
    x: torch.Tensor,
    name: str,
    n_calls: int = 100,
    warmup: int = 20,
    repeats: int = 30,
):
    """
    Benchmarks `fn(x)` repeated `n_calls` times per repeat.
    Handles CUDA synchronization for accurate GPU timing.
    """
    device = x.device

    # Warmup
    for _ in range(warmup):
        _ = fn(x)
    _sync_if_cuda(device)

    times_ms = []
    for _ in range(repeats):
        _sync_if_cuda(device)
        t0 = time.perf_counter()

        for _ in range(n_calls):
            _ = fn(x)

        _sync_if_cuda(device)
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000.0)

    t = torch.tensor(times_ms, dtype=torch.float64)

    numel = x.numel()
    total_elems_per_repeat = numel * n_calls

    mean_total_ms = t.mean().item()
    std_total_ms = t.std(unbiased=False).item()
    min_total_ms = t.min().item()
    max_total_ms = t.max().item()

    per_call_ms = mean_total_ms / n_calls
    per_elem_ns = (mean_total_ms * 1e6) / total_elems_per_repeat  # ms -> ns over all elems
    calls_per_sec = n_calls / (mean_total_ms / 1000.0)
    elems_per_sec = total_elems_per_repeat / (mean_total_ms / 1000.0)

    return {
        "name": name,
        "device": str(device),
        "shape": tuple(x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "numel": numel,
        "n_calls": n_calls,
        "warmup": warmup,
        "repeats": repeats,
        "mean_total_ms": mean_total_ms,
        "std_total_ms": std_total_ms,
        "min_total_ms": min_total_ms,
        "max_total_ms": max_total_ms,
        "per_call_ms": per_call_ms,
        "per_elem_ns": per_elem_ns,
        "calls_per_sec": calls_per_sec,
        "elems_per_sec": elems_per_sec,
    }


def print_result_block(r):
    print(f"\n[{r['name']}]")
    print(f"  Device             : {r['device']}")
    print(f"  Shape              : {r['shape']}  (numel={r['numel']})")
    print(f"  DType              : {r['dtype']}")
    print(f"  Calls / Warmup / Repeats : {r['n_calls']} / {r['warmup']} / {r['repeats']}")
    print(f"  Mean total time    : {r['mean_total_ms']:.3f} ms")
    print(f"  Std total time     : {r['std_total_ms']:.3f} ms")
    print(f"  Min / Max total    : {r['min_total_ms']:.3f} / {r['max_total_ms']:.3f} ms")
    print(f"  Mean per call      : {r['per_call_ms']:.6f} ms")
    print(f"  Mean per element   : {r['per_elem_ns']:.3f} ns")
    print(f"  Calls / sec        : {_format_num(r['calls_per_sec'])}")
    print(f"  Elements / sec     : {_format_num(r['elems_per_sec'])}")


def print_comparison_table(rows):
    # rows: list of dicts with same keys
    if not rows:
        return

    headers = [
        "device",
        "shape",
        "dtype",
        "manual_per_call_ms",
        "dist_per_call_ms",
        "speed_ratio_dist_over_manual",
        "manual_per_elem_ns",
        "dist_per_elem_ns",
        "manual_calls_per_sec",
        "dist_calls_per_sec",
    ]

    # Convert rows to printable strings
    table = []
    for row in rows:
        table.append({
            "device": row["device"],
            "shape": str(row["shape"]),
            "dtype": row["dtype"],
            "manual_per_call_ms": f"{row['manual_per_call_ms']:.6f}",
            "dist_per_call_ms": f"{row['dist_per_call_ms']:.6f}",
            "speed_ratio_dist_over_manual": f"{row['speed_ratio']:.3f}x",
            "manual_per_elem_ns": f"{row['manual_per_elem_ns']:.3f}",
            "dist_per_elem_ns": f"{row['dist_per_elem_ns']:.3f}",
            "manual_calls_per_sec": _format_num(row["manual_calls_per_sec"]),
            "dist_calls_per_sec": _format_num(row["dist_calls_per_sec"]),
        })

    # Column widths
    col_widths = {}
    for h in headers:
        max_len = len(h)
        for row in table:
            max_len = max(max_len, len(row[h]))
        col_widths[h] = max_len

    # Print
    print("\n" + "=" * 120)
    print("SUMMARY COMPARISON (prebuilt scalar dist vs manual safe)")
    print("=" * 120)

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in table:
        line = " | ".join(row[h].ljust(col_widths[h]) for h in headers)
        print(line)

    print("=" * 120)


# =========================================================
# 3) Optional sanity check for distribution correctness
# =========================================================
@torch.no_grad()
def quick_distribution_sanity(fn, x, device_name=""):
    """
    Quick mean/var check against theoretical Gumbel(0,1):
      mean = EulerGamma ≈ 0.5772156649
      var  = pi^2 / 6 ≈ 1.6449340668
    """
    y = fn(x).float()
    mean = y.mean().item()
    var = y.var(unbiased=False).item()

    theo_mean = 0.5772156649015329
    theo_var = math.pi * math.pi / 6.0

    print(f"[SANITY {device_name}] sample mean={mean:.6f} (theory {theo_mean:.6f}), "
          f"sample var={var:.6f} (theory {theo_var:.6f})")


# =========================================================
# 4) Main benchmark runner
# =========================================================
def run_device_benchmarks(
    device: torch.device,
    shapes,
    dtypes=(torch.float32,),
    n_calls=100,
    warmup=20,
    repeats=30,
    run_sanity_check=True,
):
    print(f"\n{'#' * 80}")
    print(f"Running benchmarks on device: {device}")
    print(f"{'#' * 80}")

    sampler = GumbelSampler(device=device)
    summary_rows = []

    for dtype in dtypes:
        print(f"\n--- DTYPE: {dtype} ---")

        for shape in shapes:
            x = torch.empty(shape, device=device, dtype=dtype)

            print(f"\nShape = {tuple(shape)}  | numel = {x.numel()}")

            # Optional quick sanity checks (one call only)
            if run_sanity_check:
                # Use a reasonably large tensor for stability if current shape is tiny
                quick_distribution_sanity(sample_gumbel_manual_safe, x, device_name=f"{device}-manual")
                quick_distribution_sanity(sampler.sample_like, x, device_name=f"{device}-dist")

            manual_res = benchmark_sampler(
                sample_gumbel_manual_safe,
                x,
                name="manual inverse-CDF (safe fp32 internal)",
                n_calls=n_calls,
                warmup=warmup,
                repeats=repeats,
            )
            dist_res = benchmark_sampler(
                sampler.sample_like,
                x,
                name="prebuilt scalar torch.distributions.Gumbel",
                n_calls=n_calls,
                warmup=warmup,
                repeats=repeats,
            )

            print_result_block(manual_res)
            print_result_block(dist_res)

            speed_ratio = dist_res["per_call_ms"] / manual_res["per_call_ms"]
            print(f"\n[COMPARISON] dist/manual speed ratio: {speed_ratio:.3f}x")

            summary_rows.append({
                "device": str(device),
                "shape": tuple(shape),
                "dtype": str(dtype).replace("torch.", ""),
                "manual_per_call_ms": manual_res["per_call_ms"],
                "dist_per_call_ms": dist_res["per_call_ms"],
                "speed_ratio": speed_ratio,
                "manual_per_elem_ns": manual_res["per_elem_ns"],
                "dist_per_elem_ns": dist_res["per_elem_ns"],
                "manual_calls_per_sec": manual_res["calls_per_sec"],
                "dist_calls_per_sec": dist_res["calls_per_sec"],
            })

            # Free memory between large shapes
            del x
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return summary_rows


if __name__ == "__main__":
    # -----------------------------------------------------
    # Config (edit these to match your workloads)
    # -----------------------------------------------------
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    print("[INFO] PyTorch:", torch.__version__)
    print("[INFO] CUDA available:", torch.cuda.is_available())

    # Representative shapes (dynamic-shape workloads)
    # Add/remove shapes as needed.
    shapes = [
        (32, 196),            # common token grid
        (32, 512),            # longer token vector
        (8, 12, 196, 196),    # attention map-like tensor
        (4, 12, 256, 256),    # larger attention map-like tensor
    ]

    # Dtypes to test (fp16 only if CUDA is available; CPU fp16 can be slow/unsupported in some ops)
    cpu_dtypes = (torch.float32,)
    gpu_dtypes = (torch.float32, torch.float16) if torch.cuda.is_available() else ()

    # Benchmark controls
    N_CALLS = 100
    WARMUP = 20
    REPEATS = 30

    all_summary = []

    # -----------------------------------------------------
    # CPU benchmarks
    # -----------------------------------------------------
    cpu_device = torch.device("cpu")
    all_summary.extend(
        run_device_benchmarks(
            device=cpu_device,
            shapes=shapes,
            dtypes=cpu_dtypes,
            n_calls=N_CALLS,
            warmup=WARMUP,
            repeats=REPEATS,
            run_sanity_check=True,
        )
    )

    # -----------------------------------------------------
    # GPU benchmarks (if available)
    # -----------------------------------------------------
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        print("\n[INFO] GPU:", torch.cuda.get_device_name(0))

        all_summary.extend(
            run_device_benchmarks(
                device=gpu_device,
                shapes=shapes,
                dtypes=gpu_dtypes,
                n_calls=N_CALLS,
                warmup=WARMUP,
                repeats=REPEATS,
                run_sanity_check=True,
            )
        )

    # -----------------------------------------------------
    # Summary table
    # -----------------------------------------------------
    print_comparison_table(all_summary)