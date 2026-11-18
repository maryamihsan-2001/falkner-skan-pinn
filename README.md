# falkner-skan-pinn
# Falkner-Skan Flow with Heat Transfer: PINN vs Classical Solver

Solves the Falkner-Skan boundary-layer equations with a passive temperature
field using (1) SciPy's `solve_bvp` and (2) a physics-informed neural network
in PyTorch, then compares them.

## Equations
f''' + f f'' + β(1 − f'²) = 0,  θ'' + Pr f θ' = 0
f(0)=0, f'(0)=0, f'(∞)=1, θ(0)=1, θ(∞)=0

## Run
pip install -r requirements.txt
python run_comparison.py
python prandtl_study.py

## Validation
Known values: f''(0) ≈ 0.4696 (β=0), ≈ 1.2326 (β=1).

## Results
(Insert results/comparison.png and your error table here.)

## References
- V. M. Falkner and S. W. Skan, Phil. Mag. 12 (1931) 865.
- M. Raissi, P. Perdikaris, G. E. Karniadakis, J. Comput. Phys. 378 (2019).
- Code written with AI assistance (Claude).

## Possible extensions
Magnetic (MHD) term, viscous dissipation, weak-form loss.
