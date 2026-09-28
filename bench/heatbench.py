"""heatbench: run the heat solver over many configurations and turn the results into plots.

Usage:
    python bench/heatbench.py convergence   # accuracy: error vs grid spacing, float vs double
    python bench/heatbench.py scaling       # OpenMP speedup vs number of threads
    python bench/heatbench.py size          # CPU throughput vs grid size (cache vs memory)
    python bench/heatbench.py gpu           # CUDA: correctness vs CPU, throughput vs grid size
    python bench/heatbench.py cpu           # convergence + scaling + size
    python bench/heatbench.py report        # Markdown report from whatever results exist

Options:
    --build DIR   directory containing the solver binary (default: build)

Each experiment writes results/<name>.csv and figures/<name>.png, so CPU experiments and GPU
experiments can be run on different systems (e.g. Windows for the CPU, WSL or Colab for the GPU)
and still end up in one report.
"""

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PRECISIONS = ["double", "float"]
MARKERS = {"double": "o", "float": "s"}


# --- helpers ------------------------------------------------------------------------------

def find_solver(build_dir):
    for name in ("heat.exe", "heat"):
        path = ROOT / build_dir / name
        if path.exists():
            return path
    sys.exit(f"solver not found in {build_dir}/: build it first (see README)")


def cpu_name():
    """Readable CPU model name (platform.processor() only gives the family code on Windows)."""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return out.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown GPU"


def run_solver(solver, repeats=1, **options):
    """Run the solver `repeats` times and keep the fastest run (least disturbed by the OS)."""
    args = [str(solver)]
    for key, value in options.items():
        args += ["--" + key.replace("_", "-"), str(value)]
    runs = []
    for _ in range(repeats):
        out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
        runs.append(json.loads(out.strip().splitlines()[-1]))
    return min(runs, key=lambda r: r["seconds"])


def has_cuda(solver):
    probe = subprocess.run([str(solver), "--n", "8", "--steps", "1", "--backend", "cuda"],
                           capture_output=True, text=True)
    return probe.returncode == 0


def save_csv(name, rows, machine):
    RESULTS.mkdir(exist_ok=True)
    for r in rows:
        r["machine"] = machine
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(name):
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return []
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = int(v)
            except ValueError:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def save_figure(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)


def fitted_order(hs, errors):
    """Slope of log(error) against log(h): the observed order of accuracy."""
    xs, ys = [math.log(h) for h in hs], [math.log(e) for e in errors]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def grid_mb(n, precision):
    """Memory held by the two (n+2)^2 grids."""
    return 2 * (n + 2) ** 2 * (4 if precision == "float" else 8) / 1e6


# --- experiments --------------------------------------------------------------------------

CONVERGENCE_SIZES = (16, 32, 64, 128, 256, 512)
T_END = 0.01


def convergence(solver):
    """Refine the grid at a fixed final time. The FTCS scheme should be second order in h."""
    rows = []
    for precision in PRECISIONS:
        for n in CONVERGENCE_SIZES:
            r = run_solver(solver, n=n, t_end=T_END, precision=precision, backend="omp")
            r["h"] = 1.0 / (n + 1)
            rows.append(r)
            print(f"  convergence {precision:6s} n={n:4d}  rms error = {r['rms_err']:.3e}")
    save_csv("convergence", rows, cpu_name())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for precision in PRECISIONS:
        sel = [r for r in rows if r["precision"] == precision]
        ax.loglog([r["h"] for r in sel], [r["rms_err"] for r in sel], marker=MARKERS[precision],
                  label=precision)
    ref = [r for r in rows if r["precision"] == "double"]
    h0, e0 = ref[0]["h"], ref[0]["rms_err"]
    ax.loglog([r["h"] for r in ref], [e0 * (r["h"] / h0) ** 2 for r in ref], "k--", label="slope 2")
    ax.set_xlabel("grid spacing h")
    ax.set_ylabel("RMS error vs exact solution")
    ax.set_title(f"Convergence at t = {T_END}")
    ax.legend()
    save_figure(fig, "convergence")


def scaling(solver, sizes=(512, 2048), repeats=3):
    """Speedup of the OpenMP version over the serial version as threads are added."""
    max_threads = os.cpu_count() or 1
    thread_counts = sorted({1, 2, 4, 6, 8, 10, 12, 14, 16, max_threads} & set(range(1, max_threads + 1)))
    rows = []
    for n in sizes:
        steps = max(20, int(4e8 / (n * n)))
        for precision in PRECISIONS:
            base = run_solver(solver, repeats, n=n, steps=steps, precision=precision, backend="serial")
            for p in thread_counts:
                r = run_solver(solver, repeats, n=n, steps=steps, precision=precision,
                               backend="omp", threads=p)
                r["speedup"] = base["seconds"] / r["seconds"]
                rows.append(r)
                print(f"  scaling n={n:5d} {precision:6s} threads={p:2d}  speedup = {r['speedup']:.2f}")
    save_csv("scaling", rows, cpu_name())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n, style in zip(sizes, ("-", "--")):
        for precision in PRECISIONS:
            sel = [r for r in rows if r["n"] == n and r["precision"] == precision]
            ax.plot([r["threads"] for r in sel], [r["speedup"] for r in sel], style,
                    marker=MARKERS[precision], label=f"n={n}, {precision}")
    ax.plot(thread_counts, thread_counts, "k:", label="ideal")
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("speedup over serial")
    ax.set_title(f"Strong scaling, {cpu_name()}", fontsize=10)
    ax.legend(fontsize=8)
    save_figure(fig, "scaling")


def size(solver, sizes=(128, 256, 512, 1024, 2048, 4096), repeats=3):
    """CPU throughput as the grid grows: small grids fit in cache, large ones wait on memory."""
    rows = []
    for backend in ("serial", "omp"):
        for precision in PRECISIONS:
            for n in sizes:
                steps = max(10, int(4e8 / (n * n)))
                r = run_solver(solver, repeats, n=n, steps=steps, precision=precision, backend=backend)
                r["grid_mb"] = grid_mb(n, precision)
                rows.append(r)
                print(f"  size {backend:6s} {precision:6s} n={n:5d}  {r['mlups']:9.1f} MLUPS")
    save_csv("size", rows, cpu_name())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for backend, style in (("serial", "--"), ("omp", "-")):
        for precision in PRECISIONS:
            sel = [r for r in rows if r["backend"] == backend and r["precision"] == precision]
            ax.semilogx([r["grid_mb"] for r in sel], [r["mlups"] for r in sel], style,
                        marker=MARKERS[precision], label=f"{backend}, {precision}")
    ax.set_xlabel("memory used by the two grids (MB)")
    ax.set_ylabel("million grid-point updates per second")
    ax.set_title("CPU throughput vs problem size")
    ax.legend(fontsize=8)
    save_figure(fig, "size")


def gpu(solver, sizes=(128, 256, 512, 1024, 2048, 4096, 8192), repeats=3):
    """CUDA version: first check it agrees with the CPU, then measure throughput vs grid size."""
    if not has_cuda(solver):
        sys.exit("this solver build has no CUDA back end")

    print("  checking the GPU against the CPU:")
    worst = 0.0
    for precision in PRECISIONS:
        for n in (32, 128):
            cpu = run_solver(solver, n=n, t_end=T_END, precision=precision, backend="serial")
            dev = run_solver(solver, n=n, t_end=T_END, precision=precision, backend="cuda")
            diff = abs(cpu["rms_err"] - dev["rms_err"]) / cpu["rms_err"]
            worst = max(worst, diff)
            print(f"    {precision:6s} n={n:4d}  CPU error {cpu['rms_err']:.4e}  GPU error {dev['rms_err']:.4e}")
    print(f"  largest relative difference in error: {worst:.1e}")

    rows = []
    for precision in PRECISIONS:
        for n in sizes:
            steps = max(20, min(20000, int(2e10 / (n * n))))
            r = run_solver(solver, repeats, n=n, steps=steps, precision=precision, backend="cuda")
            r["grid_mb"] = grid_mb(n, precision)
            r["check_rel_diff"] = worst
            rows.append(r)
            print(f"  gpu {precision:6s} n={n:5d}  {r['mlups']:10.1f} MLUPS  {r['gbs']:7.1f} GB/s")
    save_csv("gpu", rows, gpu_name())

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for precision in PRECISIONS:
        sel = [r for r in rows if r["precision"] == precision]
        ax.loglog([r["grid_mb"] for r in sel], [r["mlups"] for r in sel], "-",
                  marker=MARKERS[precision], label=f"GPU, {precision}")
    cpu_rows = [r for r in load_csv("size") if r["backend"] == "omp"]
    for precision in PRECISIONS:
        sel = [r for r in cpu_rows if r["precision"] == precision]
        if sel:
            ax.loglog([r["grid_mb"] for r in sel], [r["mlups"] for r in sel], "--",
                      marker=MARKERS[precision], label=f"CPU OpenMP, {precision}")
    ax.set_xlabel("memory used by the two grids (MB)")
    ax.set_ylabel("million grid-point updates per second")
    ax.set_title(f"GPU vs CPU throughput ({gpu_name()})", fontsize=10)
    ax.legend(fontsize=8)
    save_figure(fig, "gpu")


# --- report -------------------------------------------------------------------------------

def write_report():
    lines = ["# heatbench report", ""]
    conv = load_csv("convergence")
    if conv:
        lines += [f"CPU: {conv[0]['machine']}", "", "## Accuracy", "",
                  "| precision | observed order (coarse grids) | smallest RMS error |", "|---|---|---|"]
        for p in PRECISIONS:
            sel = [r for r in conv if r["precision"] == p]
            order = fitted_order([r["h"] for r in sel[:3]], [r["rms_err"] for r in sel[:3]])
            lines.append(f"| {p} | {order:.2f} | {min(r['rms_err'] for r in sel):.2e} |")
        lines += ["", "![convergence](../figures/convergence.png)", ""]
    scal = load_csv("scaling")
    if scal:
        lines += ["## OpenMP scaling", "", "| n | precision | best speedup | at threads | speedup at max threads |",
                  "|---|---|---|---|---|"]
        for n in sorted({r["n"] for r in scal}):
            for p in PRECISIONS:
                sel = [r for r in scal if r["n"] == n and r["precision"] == p]
                best = max(sel, key=lambda r: r["speedup"])
                lines.append(f"| {n} | {p} | {best['speedup']:.2f} | {best['threads']} | {sel[-1]['speedup']:.2f} |")
        lines += ["", "![scaling](../figures/scaling.png)", ""]
    if load_csv("size"):
        lines += ["## CPU throughput vs problem size", "", "![size](../figures/size.png)", ""]
    gpu_rows = load_csv("gpu")
    if gpu_rows:
        lines += [f"## GPU ({gpu_rows[0]['machine']})", "",
                  f"GPU and CPU errors agree to a relative difference of {gpu_rows[0]['check_rel_diff']:.1e}.", "",
                  "| n | grid MB (double) | double MLUPS | float MLUPS | float / double |", "|---|---|---|---|---|"]
        for n in sorted({r["n"] for r in gpu_rows}):
            d = next(r for r in gpu_rows if r["n"] == n and r["precision"] == "double")
            f = next(r for r in gpu_rows if r["n"] == n and r["precision"] == "float")
            lines.append(f"| {n} | {d['grid_mb']:.1f} | {d['mlups']:.0f} | {f['mlups']:.0f} | {f['mlups'] / d['mlups']:.2f} |")
        lines += ["", "![gpu](../figures/gpu.png)", ""]
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("report written to", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["convergence", "scaling", "size", "gpu", "cpu", "report"])
    parser.add_argument("--build", default="build", help="directory containing the solver binary")
    args = parser.parse_args()

    if args.experiment != "report":
        solver = find_solver(args.build)
        if args.experiment in ("convergence", "cpu"):
            convergence(solver)
        if args.experiment in ("scaling", "cpu"):
            scaling(solver)
        if args.experiment in ("size", "cpu"):
            size(solver)
        if args.experiment == "gpu":
            gpu(solver)
    write_report()


if __name__ == "__main__":
    main()
