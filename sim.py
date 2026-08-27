import argparse
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Parameters ────────────────────────────────────────────────────────────────

N_vals = [150, 200, 250, 300]
K_vals = [30, 60, 100, 140]

REPEATS = 300

STRATEGIC_LEVELS = [0.0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 0.9, 1.0]
POOL_FRACS = [0.25, 0.5, 0.75]

# ── Value distribution ────────────────────────────────────────────────────────

def value_distribution(n):
    return np.random.uniform(0, 1, n)

# ── Entry decisions ───────────────────────────────────────────────────────────

def enter_decision(v, mech, pool_est, strategic=False):
    if mech == "fcfs":
        p_accept = min(1.0, K / max(pool_est, 1))
    elif mech == "priority":
        p_accept = min(1.0, 1.2 * K / max(pool_est, 1))
    else:
        p_accept = 0.85 * min(1.0, K / max(pool_est, 1))

    expected_benefit = v * p_accept

    if strategic:
        cost = 0.38 if mech == "commitment" else 0.08
    else:
        cost = 0.28

    return expected_benefit > cost

# ── Dropout ───────────────────────────────────────────────────────────────────

def dropout_prob(v, strategic=False, mech="fcfs"):
    base = 1.0 - v
    if strategic:
        base += 0.2
    if mech == "commitment":
        base *= 0.4
    return float(np.clip(base, 0.0, 1.0))

# ── Allocation mechanisms ─────────────────────────────────────────────────────

def allocate_fcfs(vals, join_mask, is_strategic):
    order = np.random.permutation(N)
    offered = []

    for i in order:
        if join_mask[i]:
            offered.append(i)
            if len(offered) == K:
                break

    return [i for i in offered
            if np.random.rand() > dropout_prob(vals[i], is_strategic[i], "fcfs")]


def allocate_priority(vals, join_mask, is_strategic):
    if np.sum(join_mask) == 0:
        return []

    reliability = np.random.uniform(0, 1, N)
    scores = 0.8 * vals + 0.2 * reliability

    candidates = np.where(join_mask)[0]
    ranked = candidates[np.argsort(-scores[candidates])]
    offered = list(ranked[:K])

    return [i for i in offered
            if np.random.rand() > dropout_prob(vals[i], is_strategic[i], "priority")]


def allocate_commitment(vals, join_mask, is_strategic):
    effective_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        if join_mask[i]:
            p_drop = dropout_prob(vals[i], is_strategic[i], "commitment")
            effective_mask[i] = (np.random.rand() > p_drop)

    order = np.random.permutation(N)
    final = []

    for i in order:
        if effective_mask[i]:
            final.append(i)
            if len(final) == K:
                break

    return final

# ── Metrics ───────────────────────────────────────────────────────────────────

def efficiency(alloc, vals):
    return float(np.sum(vals[alloc])) if alloc else 0.0

def wastage(alloc):
    return (K - len(alloc)) / K

def fairness(alloc, vals):
    return float(np.mean(vals[alloc])) if alloc else 0.0

# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(strategic_fraction, pool_est):
    results = {m: [] for m in ("fcfs", "commit", "priority")}

    for _ in range(REPEATS):
        vals = value_distribution(N)
        is_strat = np.random.rand(N) < strategic_fraction

        join = {m: np.zeros(N, dtype=bool) for m in ("fcfs", "commit", "priority")}

        for i in range(N):
            join["fcfs"][i] = enter_decision(vals[i], "fcfs", pool_est, is_strat[i])
            join["commit"][i] = enter_decision(vals[i], "commitment", pool_est, is_strat[i])
            join["priority"][i] = enter_decision(vals[i], "priority", pool_est, is_strat[i])

        alloc = {
            "fcfs": allocate_fcfs(vals, join["fcfs"], is_strat),
            "commit": allocate_commitment(vals, join["commit"], is_strat),
            "priority": allocate_priority(vals, join["priority"], is_strat),
        }

        for m, a in alloc.items():
            results[m].append((efficiency(a, vals), wastage(a), fairness(a, vals)))

    out = {}
    for m in results:
        arr = np.array(results[m])
        out[m] = (np.mean(arr, axis=0), np.std(arr, axis=0) / np.sqrt(REPEATS))
    return out

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global N, K

    COLORS = {"fcfs": "#1f77b4", "commit": "#ff7f0e", "priority": "#2ca02c"}
    LABELS = {"fcfs": "FCFS", "commit": "Commitment", "priority": "Priority"}
    LINESTYLES = {0.25: "dotted", 0.5: "solid", 0.75: "dashed"}

    METRICS = [
        ("eff", "eff_se", "Efficiency (Total Value)", "Total Value"),
        ("wast", "wast_se", "Seat Wastage", "Wastage Rate"),
        ("fair", "fair_se", "Fairness (Mean Value)", "Mean Value"),
    ]

    for n in N_vals:
        N = n

        for k in K_vals:
            K = k

            print(f"\n=== N={N}, K={K} ===")

            store = {
                m: {
                    frac: {kk: [] for kk in ("eff", "wast", "fair", "eff_se", "wast_se", "fair_se")}
                    for frac in POOL_FRACS
                }
                for m in ("fcfs", "commit", "priority")
            }

            for frac in POOL_FRACS:
                pool_est = max(K, frac * N)

                for s in STRATEGIC_LEVELS:
                    sim = run_simulation(s, pool_est)

                    for m in ("fcfs", "commit", "priority"):
                        means, ses = sim[m]

                        store[m][frac]["eff"].append(means[0])
                        store[m][frac]["eff_se"].append(ses[0])
                        store[m][frac]["wast"].append(means[1])
                        store[m][frac]["wast_se"].append(ses[1])
                        store[m][frac]["fair"].append(means[2])
                        store[m][frac]["fair_se"].append(ses[2])

            # ── Plot ───────────────────────────────────────────────────────────

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for ax, (metric, se_key, title, ylabel) in zip(axes, METRICS):
                for m in ("fcfs", "commit", "priority"):
                    for frac in POOL_FRACS:
                        mu = np.array(store[m][frac][metric])
                        se = np.array(store[m][frac][se_key])

                        label = LABELS[m] if frac == POOL_FRACS[1] else None

                        ax.plot(
                            STRATEGIC_LEVELS,
                            mu,
                            color=COLORS[m],
                            linestyle=LINESTYLES[frac],
                            label=label,
                            linewidth=2
                        )

                        ax.fill_between(
                            STRATEGIC_LEVELS,
                            mu - 1.96 * se,
                            mu + 1.96 * se,
                            alpha=0.12
                        )

                ax.set_title(f"{title} (N={N}, K={K})", fontweight="bold")
                ax.set_xlabel("Fraction of Strategic Agents")
                ax.set_ylabel(ylabel)
                ax.set_xlim(0, 1)
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.legend()

            plt.tight_layout()
            fname = f"results_N{N}_K{K}.png"
            plt.savefig(fname, dpi=150)
            plt.close()

            print(f"Saved {fname}")

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()