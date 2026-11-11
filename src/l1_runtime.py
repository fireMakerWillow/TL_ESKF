"""M27 RuntimeProfiler (L1 evaluation kernel) -- NEW in v3.

The analytic FLOP and byte counts of M11 answer "how does this scale"; they do
not answer "what does it cost".  Section 4.4 of the plan is explicit that the
final resource verdict is the MEASURED runtime and the MEASURED footprint, so
this module instruments the estimators and reports both alongside M11's
analytic figures.  When the two disagree the paper has to say why -- the usual
culprits are sparsity that the FLOP count ignores and batched updates that the
FLOP count over-charges.

What is measured
----------------
    total            one Monte-Carlo trial, end to end
    predict          one IMU propagation step   (inner loop)
    update           one UWB period: update + sync + push-down + reset
    scaling in K     paired with M21's K = 8 / 10 points
    memory           tracemalloc peak, plus the exact ``nbytes`` of the
                     covariance objects the algorithm holds

Inclusive vs exclusive time.  Sections nest (``trial`` contains ``update``
contains ``sync``), so each section carries two numbers: ``total_s``, the
inclusive wall time, and ``self_s``, the inclusive time minus the time spent in
child sections.  Summing ``self_s`` over all sections gives the trial once;
summing ``total_s`` would count the inner loops several times over.  This is
what "accumulate by stack" has to mean in practice -- a single number per
section cannot be both.

Fairness (plan section 8.3).  Every method must be timed in the same process,
on the same machine, at the same BLAS thread count, after a warm-up pass that
absorbs first-touch allocation and cache effects.  :func:`fairness_report`
records the settings that were actually in force so the table can be audited
later, and :func:`warmup` performs the discard pass.

Zero-impact contract (T23).  Profiling must not perturb any number in the
TrialLog.  The profiler only reads a clock and, when memory tracking is on,
sums ``nbytes``; it never touches an estimator's state.  :class:`NullProfiler`
implements the same interface with empty bodies so a disabled run does not even
pay the dictionary lookups, and the two can be swapped without an ``if`` at
every call site.
"""

import time
import tracemalloc
from contextlib import contextmanager

import numpy as np


class _Section:
    __slots__ = ("name", "total", "child", "calls", "samples")

    def __init__(self, name):
        self.name = name
        self.total = 0.0        # inclusive seconds
        self.child = 0.0        # seconds spent inside nested sections
        self.calls = 0
        self.samples = []

    def add(self, elapsed, child, keep_samples):
        self.total += elapsed
        self.child += child
        self.calls += 1
        if keep_samples:
            self.samples.append(elapsed)


class RuntimeProfiler:
    """Wall-clock and memory profiler with nested, stack-aware sections."""

    def __init__(self, config=None, enabled=True, track_memory=False,
                 keep_samples=True):
        self.enabled = bool(enabled)
        self.track_memory = bool(track_memory)
        self.keep_samples = bool(keep_samples)
        self.repeats = int((config or {}).get("runtime_repeats", 1))
        self._sections = {}
        self._stack = []
        self._bytes = {}
        self._peak_bytes = 0
        self._tracing = False

    # ------------------------------------------------------------------
    # timing
    # ------------------------------------------------------------------
    @contextmanager
    def profile(self, section):
        """Time a named section; nested calls are attributed to the child."""
        if not self.enabled:
            yield self
            return
        entry = self._sections.get(section)
        if entry is None:
            entry = self._sections[section] = _Section(section)
        frame = [0.0]                       # child time accumulated by callees
        self._stack.append(frame)
        start = time.perf_counter()
        try:
            yield self
        finally:
            elapsed = time.perf_counter() - start
            self._stack.pop()
            entry.add(elapsed, frame[0], self.keep_samples)
            if self._stack:
                self._stack[-1][0] += elapsed

    def timed(self, section, fn, *args, **kwargs):
        """Call ``fn`` inside a profiled section and return its result."""
        with self.profile(section):
            return fn(*args, **kwargs)

    # ------------------------------------------------------------------
    # memory
    # ------------------------------------------------------------------
    def start_memory(self):
        """Begin tracemalloc tracing (interpreter-level allocation)."""
        if self.enabled and self.track_memory and not self._tracing:
            tracemalloc.start()
            self._tracing = True
        return self

    def stop_memory(self):
        """Stop tracing and return ``(current, peak)`` bytes."""
        if not self._tracing:
            return (0, 0)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._tracing = False
        self._peak_bytes = max(self._peak_bytes, int(peak))
        return (int(current), int(peak))

    def record_arrays(self, label, arrays):
        """Record the exact ``nbytes`` of the covariance objects a method holds.

        This is the algorithmic lower bound and is what the paper should quote
        next to M11's analytic float count; the tracemalloc peak additionally
        carries interpreter overhead and is reported beside it, not instead.
        """
        if not self.enabled:
            return 0
        total = 0
        for a in (arrays if isinstance(arrays, (list, tuple)) else [arrays]):
            arr = np.asarray(a)
            total += int(arr.nbytes)
        self._bytes[label] = max(self._bytes.get(label, 0), total)
        return total

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def reset(self):
        self._sections.clear()
        self._stack.clear()
        self._bytes.clear()
        self._peak_bytes = 0
        return self

    def summary(self):
        """Per-section timing plus the memory figures."""
        out = {"enabled": self.enabled, "sections": {}, "memory": {}}
        for name, entry in self._sections.items():
            self_s = max(entry.total - entry.child, 0.0)
            row = {
                "calls": entry.calls,
                "total_s": entry.total,
                "self_s": self_s,
                "mean_s": entry.total / entry.calls if entry.calls else float("nan"),
            }
            if entry.samples:
                arr = np.asarray(entry.samples, dtype=float)
                row["std_s"] = float(np.std(arr))
                row["min_s"] = float(np.min(arr))
                row["median_s"] = float(np.median(arr))
            out["sections"][name] = row
        out["memory"] = {"array_bytes": dict(self._bytes),
                         "array_bytes_total": int(sum(self._bytes.values())),
                         "tracemalloc_peak_bytes": int(self._peak_bytes)}
        return out

    def flat_row(self, prefix=""):
        """Flatten the summary into scalar columns for the TrialLog."""
        row = {}
        for name, entry in self.summary()["sections"].items():
            row[f"{prefix}rt_{name}_total_s"] = entry["total_s"]
            row[f"{prefix}rt_{name}_mean_s"] = entry["mean_s"]
            row[f"{prefix}rt_{name}_calls"] = entry["calls"]
        mem = self.summary()["memory"]
        row[f"{prefix}mem_array_bytes"] = mem["array_bytes_total"]
        row[f"{prefix}mem_peak_bytes"] = mem["tracemalloc_peak_bytes"]
        return row


class NullProfiler(RuntimeProfiler):
    """Disabled profiler: same interface, no clock reads, no bookkeeping."""

    def __init__(self, config=None):
        super().__init__(config, enabled=False, track_memory=False)

    @contextmanager
    def profile(self, section):
        yield self

    def record_arrays(self, label, arrays):
        return 0


def make_profiler(config):
    """Factory honouring ``config['enable_runtime_profile']``."""
    if config.get("enable_runtime_profile", True):
        return RuntimeProfiler(config, enabled=True, track_memory=True)
    return NullProfiler(config)


def warmup(fn, *args, **kwargs):
    """Run ``fn`` once and discard the result (first-touch / cache effects)."""
    fn(*args, **kwargs)


def repeat_timing(fn, repeats=3, warm=True):
    """Time ``fn`` ``repeats`` times and return the dispersion statistics.

    Python and the OS make a single wall-clock reading nearly meaningless, so
    the plan requires ``runtime_repeats >= 3``.  The MINIMUM is reported next to
    the mean: it is the estimate least polluted by scheduler noise, while the
    standard deviation says whether the environment was quiet enough to trust
    the comparison at all.
    """
    if warm:
        warmup(fn)
    samples = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples, dtype=float)
    return {"mean_s": float(arr.mean()), "std_s": float(arr.std()),
            "min_s": float(arr.min()), "max_s": float(arr.max()),
            "repeats": int(repeats), "samples": samples}


def scaling_report(times_by_K):
    """Fit ``t ~ K^alpha`` over the measured entity counts.

    Two points (K = 8 and K = 10) give one exponent; it is reported as an
    indication, not as an established growth law, and the raw times are kept.
    """
    ks = np.asarray(sorted(times_by_K), dtype=float)
    ts = np.asarray([times_by_K[int(k)] for k in ks], dtype=float)
    out = {"K": ks.tolist(), "time_s": ts.tolist(), "exponent": float("nan")}
    good = (ks > 0) & (ts > 0)
    if int(good.sum()) >= 2:
        slope = np.polyfit(np.log(ks[good]), np.log(ts[good]), 1)[0]
        out["exponent"] = float(slope)
        out["exponent_is_two_point"] = bool(int(good.sum()) == 2)
    return out


def fairness_report():
    """Record the environment settings the timings were taken under."""
    import os
    import platform
    threads = {var: os.environ.get(var) for var in
               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")}
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas_threads_env": threads,
        "perf_counter_resolution_s": time.get_clock_info("perf_counter").resolution,
    }


if __name__ == "__main__":
    from config import default_config, validate_config

    cfg = default_config(); validate_config(cfg)

    # ---- 1) a stub of known duration is timed within tolerance ----
    prof = RuntimeProfiler(cfg, enabled=True, track_memory=True)
    with prof.profile("sleep"):
        time.sleep(0.05)
    s = prof.summary()["sections"]["sleep"]
    assert 0.045 < s["total_s"] < 0.12, s
    assert s["calls"] == 1

    # ---- 2) nesting: children sum to no more than the parent, and the
    #         parent's SELF time excludes them ----
    prof.reset()
    with prof.profile("trial"):
        for _ in range(3):
            with prof.profile("predict"):
                time.sleep(0.005)
        with prof.profile("update"):
            time.sleep(0.01)
            with prof.profile("sync"):
                time.sleep(0.005)
        time.sleep(0.005)
    sec = prof.summary()["sections"]
    assert sec["predict"]["calls"] == 3
    child_total = sec["predict"]["total_s"] + sec["update"]["total_s"]
    assert child_total <= sec["trial"]["total_s"] + 1e-6, (child_total, sec)
    assert sec["trial"]["self_s"] < sec["trial"]["total_s"]
    assert abs(sec["trial"]["self_s"] - (sec["trial"]["total_s"] - child_total)) < 1e-9
    # the grandchild is charged to update's child time, not to trial's
    assert sec["update"]["self_s"] < sec["update"]["total_s"]
    assert sec["sync"]["total_s"] <= sec["update"]["total_s"] + 1e-6
    # summing SELF over all sections reconstructs the trial exactly once
    self_sum = sum(v["self_s"] for v in sec.values())
    assert abs(self_sum - sec["trial"]["total_s"]) < 1e-6, (self_sum, sec)

    # ---- 3) repeated sections keep per-call statistics ----
    assert sec["predict"]["mean_s"] > 0 and "std_s" in sec["predict"]
    assert sec["predict"]["min_s"] <= sec["predict"]["mean_s"]

    # ---- 4) memory: nbytes is exact, tracemalloc peak is an upper figure ----
    prof.reset()
    prof.start_memory()
    P = np.zeros((120, 120))                       # 15K x 15K for K = 8
    recorded = prof.record_arrays("P_x", P)
    assert recorded == P.nbytes == 120 * 120 * 8
    current, peak = prof.stop_memory()
    assert peak >= P.nbytes
    mem = prof.summary()["memory"]
    assert mem["array_bytes_total"] == P.nbytes
    assert mem["tracemalloc_peak_bytes"] >= P.nbytes
    # recording the same label twice keeps the maximum, not the sum
    prof.record_arrays("P_x", P)
    assert prof.summary()["memory"]["array_bytes_total"] == P.nbytes

    # ---- 5) T23: profiling changes no computed value ----
    def workload(profiler):
        rng = np.random.default_rng(12345)
        acc = 0.0
        with profiler.profile("outer"):
            for _ in range(5):
                with profiler.profile("inner"):
                    M = rng.standard_normal((20, 20))
                    acc += float(np.linalg.det(M @ M.T + np.eye(20)))
        return acc

    on = workload(RuntimeProfiler(cfg, enabled=True))
    off = workload(NullProfiler(cfg))
    assert on == off, (on, off)                    # bit-identical, not "close"
    null = NullProfiler(cfg)
    workload(null)
    assert null.summary()["sections"] == {}
    assert null.record_arrays("x", np.zeros(10)) == 0
    assert null.flat_row()["mem_array_bytes"] == 0

    # a disabled profiler is also cheap: no per-call bookkeeping at all
    disabled = RuntimeProfiler(cfg, enabled=False)
    with disabled.profile("nothing"):
        pass
    assert disabled.summary()["sections"] == {}

    # ---- 6) the factory honours the config switch ----
    assert isinstance(make_profiler(cfg), RuntimeProfiler)
    off_cfg = default_config(); off_cfg["enable_runtime_profile"] = False
    validate_config(off_cfg)
    assert isinstance(make_profiler(off_cfg), NullProfiler)

    # ---- 7) repeat timing and the K-scaling fit ----
    def cubic():
        A = np.random.default_rng(0).standard_normal((60, 60))
        np.linalg.inv(A @ A.T + np.eye(60))

    rt = repeat_timing(cubic, repeats=cfg["runtime_repeats"])
    assert rt["repeats"] == cfg["runtime_repeats"] >= 3
    assert rt["min_s"] <= rt["mean_s"] <= rt["max_s"]
    fit = scaling_report({8: 1.0, 10: (10 / 8) ** 2})
    assert abs(fit["exponent"] - 2.0) < 1e-9 and fit["exponent_is_two_point"]

    # ---- 8) the flat TrialLog row and the fairness record ----
    row = prof.flat_row()
    assert "mem_array_bytes" in row and "mem_peak_bytes" in row
    fair = fairness_report()
    assert "numpy" in fair and "blas_threads_env" in fair

    print("[M27 RuntimeProfiler] OK  (nested self-time exact to %.1e s, "
          "profiling bit-identical on/off, peak=%.1f kB vs nbytes=%.1f kB)"
          % (abs(self_sum - sec["trial"]["total_s"]),
             mem["tracemalloc_peak_bytes"] / 1024.0, P.nbytes / 1024.0))
