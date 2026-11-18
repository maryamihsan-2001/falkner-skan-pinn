import os
import matplotlib.pyplot as plt
from solvers import solve_reference, wall_values

os.makedirs("results", exist_ok=True)
Prs = [0.7, 1.0, 3.0, 7.0, 10.0]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for beta, ls in [(0.0, "-"), (1.0, "--")]:
    nus = []
    for Pr in Prs:
        sol = solve_reference(beta, Pr)
        nus.append(wall_values(sol)[1])
        if beta == 0.0:
            x = sol.x
            ax[0].plot(x, sol.sol(x)[3], label=f"Pr={Pr}")
    ax[1].plot(Prs, nus, "o" + ls, label=f"β={beta}")
    print(f"beta={beta}", [round(n, 4) for n in nus])
ax[0].set_xlabel("η"); ax[0].set_ylabel("θ(η)"); ax[0].legend()
ax[0].set_title("Thermal boundary layer (β=0)")
ax[1].set_xlabel("Pr"); ax[1].set_ylabel("-θ'(0)"); ax[1].legend()
plt.tight_layout()
plt.savefig("results/prandtl_study.png", dpi=200)
