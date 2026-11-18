import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_bvp

torch.set_default_dtype(torch.float64)


# ---------- Classical reference solver ----------
def solve_reference(beta, Pr, L=8.0):
    """y = [f, f', f'', theta, theta']"""
    def rhs(x, y):
        f, f1, f2, th, th1 = y
        return np.vstack([f1, f2,
                          -(f * f2 + beta * (1 - f1**2)),
                          th1,
                          -Pr * f * th1])

    def bc(ya, yb):
        return np.array([ya[0], ya[1], yb[1] - 1, ya[3] - 1, yb[3]])

    x = np.linspace(0, L, 200)
    e = np.exp(-x)
    y0 = np.vstack([x - 1 + e, 1 - e, e, e, -e])
    sol = solve_bvp(rhs, bc, x, y0, tol=1e-8, max_nodes=100000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def wall_values(sol):
    y = sol.sol(0.0)
    return y[2], -y[4]          # f''(0), Nusselt-type -theta'(0)


# ---------- PINN ----------
class MLP(nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        layers, i = [], 1
        for _ in range(depth):
            layers += [nn.Linear(i, width), nn.Tanh()]
            i = width
        layers.append(nn.Linear(i, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


class PINN(nn.Module):
    def __init__(self, beta, Pr, L=8.0):
        super().__init__()
        self.beta, self.Pr, self.L = beta, Pr, L
        self.nf, self.nt = MLP(), MLP()

    def fields(self, x):
        z = x / self.L
        f = z**2 * self.nf(z) * self.L        # f(0)=0, f'(0)=0 built in
        th = (1 - z) + z * (1 - z) * self.nt(z)  # theta(0)=1, theta(L)=0
        return f, th

    def loss(self, x, xL):
        f, th = self.fields(x)
        f1 = grad(f, x); f2 = grad(f1, x); f3 = grad(f2, x)
        t1 = grad(th, x); t2 = grad(t1, x)
        r1 = f3 + f * f2 + self.beta * (1 - f1**2)
        r2 = t2 + self.Pr * f * t1
        fL, _ = self.fields(xL)
        bc = (grad(fL, xL) - 1) ** 2          # f'(L) = 1 (soft)
        return (r1**2).mean() + (r2**2).mean() + 10 * bc.sum()

    def predict(self, n=400):
        x = torch.linspace(0, self.L, n).reshape(-1, 1).requires_grad_(True)
        f, th = self.fields(x)
        f1 = grad(f, x); f2 = grad(f1, x); t1 = grad(th, x)
        g = lambda t: t.detach().numpy().ravel()
        return dict(x=g(x), f=g(f), f1=g(f1), f2=g(f2), th=g(th), th1=g(t1))


def train_pinn(beta, Pr, L=8.0, adam=6000, lbfgs=300, seed=0):
    torch.manual_seed(seed)
    m = PINN(beta, Pr, L)
    x = torch.linspace(0, L, 400).reshape(-1, 1).requires_grad_(True)
    xL = torch.tensor([[L]], requires_grad=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, 2000, 0.5)
    for i in range(adam):
        opt.zero_grad()
        l = m.loss(x, xL)
        l.backward()
        opt.step(); sch.step()
        if i % 1000 == 0:
            print(f"Adam {i:5d}  loss {l.item():.3e}")
    lb = torch.optim.LBFGS(m.parameters(), lr=0.5, max_iter=lbfgs,
                           tolerance_grad=1e-12, tolerance_change=1e-14,
                           line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        l = m.loss(x, xL)
        l.backward()
        return l
    lb.step(closure)
    print(f"Final loss {m.loss(x, xL).item():.3e}")
    return m


def pinn_wall_values(m):
    o = m.predict(n=2)
    return o["f2"][0], -o["th1"][0]
