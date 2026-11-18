import os
import matplotlib.pyplot as plt
from solvers import solve_reference, wall_values, train_pinn, pinn_wall_values

os.makedirs("results", exist_ok=True)
cases = [(0.0, 0.71), (1.0, 0.71)]       # (beta, Pr): flat plate, stagnation point

print(f"{'beta':>5}{'Pr':>6}{'Quantity':>12}{'Reference':>12}{'PINN':>12}{'Err %':>8}")
fig, axes = plt.subplots(len(cases), 2, figsize=(10, 7))
for row, (beta, Pr) in enumerate(cases):
    sol = solve_reference(beta, Pr)
    m = train_pinn(beta, Pr)
    out = m.predict()
    ref_f, ref_n = wall_values(sol)
    nn_f, nn_n = pinn_wall_values(m)
    for name, a, b in [("f''(0)", ref_f, nn_f), ("-theta'(0)", ref_n, nn_n)]:
        print(f"{beta:>5}{Pr:>6}{name:>12}{a:>12.5f}{b:>12.5f}{abs((a-b)/a)*100:>8.2f}")
    r = sol.sol(out["x"])
    for col, (key, idx, lab) in enumerate([("f1", 1, "f'(η)"), ("th", 3, "θ(η)")]):
        ax = axes[row, col]
        ax.plot(out["x"], r[idx], "k-", label="solve_bvp")
        ax.plot(out["x"], out[key], "r--", label="PINN")
        ax.set_xlabel("η"); ax.set_ylabel(lab)
        ax.set_title(f"β={beta}, Pr={Pr}"); ax.legend()
plt.tight_layout()
plt.savefig("results/comparison.png", dpi=200)
