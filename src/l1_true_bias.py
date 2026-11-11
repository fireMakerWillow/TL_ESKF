"""M6 TrueBiasProcess (L1 basic capability) -- truth-side sensor physics.

The TRUE accelerometer and gyroscope biases are generated here as a random walk

    b(0) ~ N(0, sigma_init^2 I),      b(t + dt) = b(t) + sqrt(dt) sigma_rw n,

driven by ``true_bias_rw_acc`` / ``true_bias_rw_gyro`` (units per sqrt(s)).

Separation contract (plan section 1.3, checked by M20's T-separation test).
The truth parameters ``true_bias_rw_*`` and the filter parameters
``filter_sigma_*_bias`` are DIFFERENT keys on purpose.  A filter that is handed
its own truth generator would silently report an optimistic NEES; keeping the
two families apart is what makes the consistency verdict meaningful.  Nothing in
this module may ever be imported by an estimator -- only M13 (the simulator) is
allowed to touch it.

The process is driven by an explicit ``numpy.random.Generator`` supplied by M13,
so a given seed reproduces the truth bit-for-bit at every (v, T) grid point.
"""

import numpy as np


class TrueBiasProcess:
    """Per-entity true IMU bias random walk (truth side only)."""

    def __init__(self, config):
        self.K = int(config["num_entities"])
        self.sigma_init_a = float(config["true_bias_init_std_acc"])
        self.sigma_init_g = float(config["true_bias_init_std_gyro"])
        self.rw_a = float(config["true_bias_rw_acc"])
        self.rw_g = float(config["true_bias_rw_gyro"])
        self.b_a = np.zeros((self.K, 3))
        self.b_g = np.zeros((self.K, 3))
        self._rng = None

    def reset(self, rng):
        """Draw the initial biases from the truth RNG."""
        self._rng = rng
        self.b_a = self.sigma_init_a * rng.standard_normal((self.K, 3))
        self.b_g = self.sigma_init_g * rng.standard_normal((self.K, 3))
        return self.b_a.copy(), self.b_g.copy()

    def step(self, dt):
        """Advance the random walk by ``dt`` and return the new biases."""
        if self._rng is None:
            raise RuntimeError("TrueBiasProcess.reset(rng) must be called first")
        s = np.sqrt(float(dt))
        self.b_a = self.b_a + s * self.rw_a * self._rng.standard_normal((self.K, 3))
        self.b_g = self.b_g + s * self.rw_g * self._rng.standard_normal((self.K, 3))
        return self.b_a.copy(), self.b_g.copy()

    def current(self):
        return self.b_a.copy(), self.b_g.copy()


def corrupt_imu(f_b_true, w_b_true, b_a, b_g, sigma_acc, sigma_gyro, dt, rng):
    """Turn the TRUE (f^b, omega^b) into a measured IMU sample.

    Discretised white noise: a density ``sigma`` (per sqrt(Hz)) integrated over a
    step ``dt`` has standard deviation ``sigma / sqrt(dt)``, which is the form
    that makes the propagated covariance independent of the sampling rate.
    """
    f_b_true = np.asarray(f_b_true, dtype=float).ravel()
    w_b_true = np.asarray(w_b_true, dtype=float).ravel()
    scale = 1.0 / np.sqrt(float(dt))
    f_meas = f_b_true + np.asarray(b_a, dtype=float).ravel() \
        + sigma_acc * scale * rng.standard_normal(3)
    w_meas = w_b_true + np.asarray(b_g, dtype=float).ravel() \
        + sigma_gyro * scale * rng.standard_normal(3)
    return f_meas, w_meas


def separation_report(config):
    """Truth vs filter noise parameters, for the M20 separation check (T-sep).

    A ``ratio`` far from 1 is expected and healthy; a ratio of exactly 1 on ALL
    channels means somebody wired the filter to its own truth generator.
    """
    pairs = {
        "acc_bias_rw": (float(config["true_bias_rw_acc"]),
                        float(config["filter_sigma_acc_bias"])),
        "gyro_bias_rw": (float(config["true_bias_rw_gyro"]),
                         float(config["filter_sigma_gyro_bias"])),
    }
    out = {}
    for name, (truth, filt) in pairs.items():
        out[name] = {"truth": truth, "filter": filt,
                     "ratio": (truth / filt) if filt else float("inf")}
    out["matched"] = all(abs(v["ratio"] - 1.0) < 1e-12 for k, v in out.items()
                         if k != "matched")
    return out


if __name__ == "__main__":
    from config import default_config, validate_config

    cfg = default_config(); validate_config(cfg)
    K = cfg["num_entities"]

    # ---- 1) reproducibility: the same seed gives the same truth ----
    a = TrueBiasProcess(cfg); a.reset(np.random.default_rng(7))
    b = TrueBiasProcess(cfg); b.reset(np.random.default_rng(7))
    for _ in range(50):
        ba, bg = a.step(0.005)
        bb, bgb = b.step(0.005)
        assert np.allclose(ba, bb) and np.allclose(bg, bgb)
    c = TrueBiasProcess(cfg); c.reset(np.random.default_rng(8))
    assert not np.allclose(c.current()[0], a.current()[0])

    # ---- 2) initial spread matches the configured std devs ----
    samples = []
    for s in range(400):
        proc = TrueBiasProcess(cfg)
        ba, _ = proc.reset(np.random.default_rng(1000 + s))
        samples.append(ba.ravel())
    emp = float(np.std(np.concatenate(samples)))
    assert abs(emp - cfg["true_bias_init_std_acc"]) < 0.1 * cfg["true_bias_init_std_acc"]

    # ---- 3) the random walk variance grows linearly in t ----
    dt, n = 0.01, 400
    drifts = []
    for s in range(300):
        proc = TrueBiasProcess(cfg)
        b0, _ = proc.reset(np.random.default_rng(500 + s))
        for _ in range(n):
            proc.step(dt)
        drifts.append((proc.current()[0] - b0).ravel())
    emp_rw = float(np.std(np.concatenate(drifts)))
    expected_rw = cfg["true_bias_rw_acc"] * np.sqrt(n * dt)
    assert abs(emp_rw - expected_rw) < 0.15 * expected_rw, (emp_rw, expected_rw)

    # ---- 4) stepping before reset is an error (no silent zero bias) ----
    try:
        TrueBiasProcess(cfg).step(0.01)
        raise AssertionError("stepping an unreset process was allowed")
    except RuntimeError:
        pass

    # ---- 5) corrupt_imu adds bias exactly and noise with the right scale ----
    rng = np.random.default_rng(3)
    f_true = np.array([0.1, 0.2, 9.8]); w_true = np.array([0.0, 0.0, 0.07])
    ba = np.array([0.02, -0.01, 0.03]); bg = np.array([1e-3, 0.0, -2e-3])
    f_m, w_m = corrupt_imu(f_true, w_true, ba, bg, 0.0, 0.0, 0.005, rng)
    assert np.allclose(f_m, f_true + ba) and np.allclose(w_m, w_true + bg)
    dt = 0.005
    devs = np.stack([corrupt_imu(f_true, w_true, ba, bg,
                                 cfg["sigma_acc"], cfg["sigma_gyro"], dt, rng)[0]
                     - f_true - ba for _ in range(4000)])
    assert abs(float(np.std(devs)) - cfg["sigma_acc"] / np.sqrt(dt)) \
        < 0.06 * cfg["sigma_acc"] / np.sqrt(dt)

    # ---- 6) truth and filter parameters live in SEPARATE keys.  The default
    #        config happens to tune them to the same numeric value (a matched
    #        filter), which is legitimate; what must never happen is that a
    #        change to one silently moves the other.
    rep = separation_report(cfg)
    assert rep["matched"], rep            # default = matched tuning
    mis = default_config(); mis["filter_sigma_acc_bias"] = 5e-3
    validate_config(mis)
    assert mis["true_bias_rw_acc"] == cfg["true_bias_rw_acc"]
    rep2 = separation_report(mis)
    assert not rep2["matched"] and abs(rep2["acc_bias_rw"]["ratio"] - 0.2) < 1e-12
    proc_a = TrueBiasProcess(cfg); proc_a.reset(np.random.default_rng(11))
    proc_b = TrueBiasProcess(mis); proc_b.reset(np.random.default_rng(11))
    for _ in range(20):
        proc_a.step(0.01); proc_b.step(0.01)
    assert np.allclose(proc_a.current()[0], proc_b.current()[0]), \
        "retuning the FILTER must not perturb the TRUTH stream"

    print("[M6 TrueBiasProcess] OK  (init std %.4f vs %.4f, RW drift %.2e vs "
          "%.2e, truth/filter ratio acc=%.2f gyro=%.2f)"
          % (emp, cfg["true_bias_init_std_acc"], emp_rw, expected_rw,
             rep["acc_bias_rw"]["ratio"], rep["gyro_bias_rw"]["ratio"]))
