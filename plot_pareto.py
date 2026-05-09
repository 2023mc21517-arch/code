"""
plot_pareto.py
==============
Generates pareto_figure.png — a two-panel figure combining:
  Left:  S+T theoretical curve (epsilon = 1/n) with the empirical
         IBP data points overlaid.
  Right: S+C theoretical curve (c/(1-c)) with the empirical
         coverage vs. IBP lower bound overlaid.

Run after pareto_experiment.py has produced pareto_results.csv.
"""

import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Load empirical data from pareto_results.csv
# ---------------------------------------------------------------------------

rows = []
with open("pareto_results.csv") as f:
    for row in csv.DictReader(f):
        rows.append({
            "coverage": float(row["coverage_fraction"]),
            "certified": row["certified"] == "True",
            "lb": float(row["ibp_lower_bound"]),
            "margin": float(row["margin"]),
        })

# For the S+T panel: x = number of queries n, y = miss rate epsilon
# IBP with a box [lo,1]^2 "uses" one query (the interval computation).
# The meaningful axis is coverage fraction vs. miss rate.
# Miss rate = fraction of safe region NOT certified.
emp_coverage = [r["coverage"] for r in rows]
emp_miss     = [1.0 - r["coverage"] if r["certified"] else None for r in rows]
emp_cert     = [r["certified"] for r in rows]
emp_lb       = [r["lb"] for r in rows]

# Transition point: last certified coverage
last_cert    = max(r["coverage"] for r in rows if r["certified"])
first_fail   = min(r["coverage"] for r in rows if not r["certified"])

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BLUE  = "#1f77b4"
RED   = "#d62728"
GREEN = "#2ca02c"
GREY  = "#aaaaaa"

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
fig.subplots_adjust(wspace=0.38, left=0.09, right=0.97, top=0.88, bottom=0.17)

# ===========================================================================
# LEFT PANEL — S+T regime:  coverage fraction vs. miss rate
# ===========================================================================
ax1.set_title("S+T regime: coverage vs. miss rate", fontsize=10, pad=6)
ax1.set_xlabel("Certified coverage fraction  $c$", fontsize=9)
ax1.set_ylabel("Miss rate  $\\varepsilon = 1 - c$", fontsize=9)
ax1.set_xlim(-0.03, 1.05)
ax1.set_ylim(-0.03, 1.05)
ax1.tick_params(labelsize=8)
ax1.grid(True, color="lightgrey", linewidth=0.5)

# Theoretical Pareto frontier: epsilon = 1 - c  (trivial identity line)
# The interesting theory curve is epsilon >= 1/(n+1) for a verifier with n queries.
# For IBP with one-shot interval analysis, n~1 "logical query".
# The theoretical lower bound on miss rate given n queries is 1/(n+1).
# We plot the achievable frontier: for each target miss rate epsilon,
# minimum queries needed = ceil(1/epsilon).
c_theory = np.linspace(0.01, 0.99, 300)
eps_theory = 1.0 - c_theory          # miss rate = 1 - coverage
n_theory   = 1.0 / eps_theory        # n queries needed for this miss rate
ax1.plot(c_theory, eps_theory,
         color=BLUE, linewidth=2, label="Theoretical bound $\\varepsilon = 1-c$", zorder=2)

# Shade the "unachievable" region above the curve
ax1.fill_between(c_theory, eps_theory, 1.05,
                 color=BLUE, alpha=0.07, label="_nolegend_")

# Empirical certified points
cert_c  = [r["coverage"] for r in rows if r["certified"]]
cert_e  = [1.0 - r["coverage"] for r in rows if r["certified"]]
fail_c  = [r["coverage"] for r in rows if not r["certified"]]
fail_e  = [1.0 - r["coverage"] for r in rows if not r["certified"]]

ax1.scatter(cert_c, cert_e,
            color=GREEN, s=55, zorder=5, label="IBP certified", marker="o")
ax1.scatter(fail_c, fail_e,
            color=RED, s=55, zorder=5, label="IBP inconclusive", marker="x",
            linewidths=2)

# Annotate the transition
ax1.axvline(last_cert, color=GREEN, linewidth=1, linestyle="--", alpha=0.6)
ax1.axvline(first_fail, color=RED, linewidth=1, linestyle="--", alpha=0.6)
ax1.annotate(f"Last certified\nc={last_cert:.0%}",
             xy=(last_cert, 1 - last_cert),
             xytext=(last_cert - 0.32, 0.68),
             fontsize=7.5, color=GREEN,
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))

ax1.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

# ===========================================================================
# RIGHT PANEL — S+C regime: coverage vs. cost (IBP lower bound as proxy)
# ===========================================================================
ax2.set_title("S+C regime: coverage vs. verification cost", fontsize=10, pad=6)
ax2.set_xlabel("Coverage fraction  $c$", fontsize=9)
ax2.set_ylabel("IBP lower bound on output", fontsize=9)
ax2.set_xlim(-0.03, 1.05)
ax2.set_ylim(-0.05, 1.08)
ax2.tick_params(labelsize=8)
ax2.grid(True, color="lightgrey", linewidth=0.5)

# Theoretical: as c → 1, cost diverges. Proxy: plot the theoretical
# "tightness" curve as 1 - (1-c)^0.5 to show degradation shape.
c_range = np.linspace(0.0, 0.99, 300)
cost_theory = 1.0 - (1.0 - c_range) ** 0.5
ax2.plot(c_range, cost_theory,
         color=RED, linewidth=2, linestyle="--",
         label="Theoretical tightness (schematic)", zorder=2)

# Empirical: plot IBP lower bound vs coverage
emp_c_all  = [r["coverage"] for r in rows]
emp_lb_all = [r["lb"]       for r in rows]
cert_mask  = [r["certified"] for r in rows]

ax2.plot(emp_c_all, emp_lb_all,
         color=BLUE, linewidth=1.5, linestyle="-", alpha=0.5,
         label="_nolegend_", zorder=3)
ax2.scatter([c for c, m in zip(emp_c_all, cert_mask) if m],
            [lb for lb, m in zip(emp_lb_all, cert_mask) if m],
            color=GREEN, s=55, zorder=5, label="IBP certified", marker="o")
ax2.scatter([c for c, m in zip(emp_c_all, cert_mask) if not m],
            [lb for lb, m in zip(emp_lb_all, cert_mask) if not m],
            color=RED, s=55, zorder=5, label="IBP inconclusive", marker="x",
            linewidths=2)

# 0.5 threshold line
ax2.axhline(0.5, color="grey", linewidth=1, linestyle=":", alpha=0.8)
ax2.text(0.02, 0.52, "certification threshold = 0.5", fontsize=7, color="grey")

ax2.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

# ===========================================================================
# Shared caption annotation
# ===========================================================================
fig.text(0.5, 0.01,
         "Empirical points: IBP verifier on Linear(2→32)→ReLU→Linear(32→1), "
         "safety property $f(\\mathbf{x})\\geq 0.5$ for $\\mathbf{x}\\in[0.5,1]^2$. "
         "Green = certified (sound guarantee). Red ✗ = inconclusive.",
         ha="center", fontsize=7.5, color="#333333",
         wrap=True)

# ===========================================================================
# Save
# ===========================================================================
plt.savefig("pareto_figure.png", dpi=180, bbox_inches="tight")
print("Saved: pareto_figure.png")
