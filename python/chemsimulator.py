# %% [markdown]
# # ChemSimulator: Python companion
# **Author:** Subhajit Majumder
#
# This script reproduces the calculations and data-science tools of the ChemSimulator web app
# (`index.html`) in Python, using NumPy, SciPy, scikit-learn and Matplotlib.
#
# 1. Unit operation models: pipe flow, heat exchanger (LMTD), CSTR/PFR reactor, mixer balance
# 2. Kinetics fitting: log-linear regression and nonlinear least squares
# 3. Neural-network surrogate trained on noisy data, compared with a linear baseline
# 4. Monte Carlo uncertainty analysis
# 5. Techno-economic pipe diameter optimisation
# 6. Anomaly and fouling detection with a residual control chart
#
# Run as a script (`python chemsimulator.py`) or open as a notebook (`chemsimulator.ipynb`).

# %%
import os
import numpy as np
import matplotlib
try:
    get_ipython()
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # save figures to files when run as a plain script
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)

# %% [markdown]
# ## 1. Unit operation models (deterministic physics)

# %%
def pipe_calc(q_m3h, d_mm, length_m, rough_mm=0.045, rho=998.0, mu_mpas=1.0):
    """Darcy-Weisbach pressure drop. Friction factor: 64/Re (laminar) or Swamee-Jain (turbulent)."""
    q_m3h, d_mm, length_m = map(np.asarray, (q_m3h, d_mm, length_m))
    d = d_mm / 1000.0
    v = q_m3h / 3600.0 / (np.pi * d**2 / 4)
    re = rho * v * d / (mu_mpas / 1000.0)
    sj = 0.25 / np.log10(rough_mm / 1000 / (3.7 * d) + 5.74 / np.maximum(re, 1) ** 0.9) ** 2
    f = np.where(re < 2300, 64 / np.maximum(re, 1e-9), sj)
    dp_kpa = f * length_m / d * rho * v**2 / 2 / 1000.0
    return dict(v=v, re=re, f=f, dp=dp_kpa)

def lmtd(a, b):
    return a if abs(a - b) < 1e-9 else (a - b) / np.log(a / b)

def heat_exchanger(t_hot_in, t_hot_out, t_cold_in, t_cold_out, duty_kw, u_w_m2k):
    """Required area for counter-current and parallel flow using the LMTD method."""
    lm_c = lmtd(t_hot_in - t_cold_out, t_hot_out - t_cold_in)
    lm_p = lmtd(t_hot_in - t_cold_in, t_hot_out - t_cold_out)
    return dict(lmtd_counter=lm_c, area_counter=duty_kw * 1000 / (u_w_m2k * lm_c),
                lmtd_parallel=lm_p, area_parallel=duty_kw * 1000 / (u_w_m2k * lm_p))

def reactor_volumes(k_per_min, v0_l_min, conversion):
    """First-order, constant density: CSTR V = v0 X / (k (1-X)), PFR V = -v0 ln(1-X) / k."""
    return dict(cstr=v0_l_min * conversion / (k_per_min * (1 - conversion)),
                pfr=-v0_l_min * np.log(1 - conversion) / k_per_min)

def mixer(flows, fractions):
    flows, fractions = np.asarray(flows, float), np.asarray(fractions, float)
    total = flows.sum()
    return dict(total=total, x_out=(flows * fractions).sum() / total)

print("Pipe (30 m3/h, 80 mm, 120 m, water):", {k: round(float(v), 3) for k, v in pipe_calc(30, 80, 120).items()})
print("Exchanger:", {k: round(float(v), 2) for k, v in heat_exchanger(150, 90, 30, 70, 500, 600).items()})
print("Reactor (k=0.25, 40 L/min, X=0.8):", {k: round(float(v), 1) for k, v in reactor_volumes(0.25, 40, 0.8).items()})
print("Mixer:", {k: round(float(v), 3) for k, v in mixer([500, 300, 200], [0.9, 0.4, 0.05]).items()})

# %% [markdown]
# ## 2. Kinetics fitting (regression)
# First-order model `C = C0 exp(-k t)` fitted by log-linear regression, refined by nonlinear least squares,
# and compared against a second-order model using R^2 of the linearised forms.

# %%
t_data = np.array([0, 2, 4, 6, 8, 10, 12, 15, 20], float)
c_data = np.array([2.02, 1.25, 0.71, 0.46, 0.262, 0.171, 0.095, 0.050, 0.0142])

def fit_kinetics(t, c):
    slope, intercept = np.polyfit(t, np.log(c), 1)
    r2_first = r2_score(np.log(c), slope * t + intercept)
    s2, i2 = np.polyfit(t, 1 / c, 1)
    r2_second = r2_score(1 / c, s2 * t + i2)
    model = lambda t, c0, k: c0 * np.exp(-k * t)
    (c0, k), cov = curve_fit(model, t, c, p0=[np.exp(intercept), -slope])
    resid = c - model(t, c0, k)
    return dict(k=k, c0=c0, k_se=np.sqrt(cov[1, 1]), r2=r2_score(c, model(t, c0, k)),
                rmse=np.sqrt(np.mean(resid**2)), half_life=np.log(2) / k,
                r2_first_linear=r2_first, r2_second_linear=r2_second, resid=resid, model=model)

fit = fit_kinetics(t_data, c_data)
print(f"k = {fit['k']:.4f} +/- {fit['k_se']:.4f} 1/min, C0 = {fit['c0']:.3f} mol/L, R2 = {fit['r2']:.4f}, "
      f"half-life = {fit['half_life']:.2f} min")
print("Better model by R2:", "first-order" if fit["r2_first_linear"] >= fit["r2_second_linear"] else "second-order")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
tt = np.linspace(t_data.min(), t_data.max(), 200)
ax[0].plot(tt, fit["model"](tt, fit["c0"], fit["k"]), label="first-order fit")
ax[0].scatter(t_data, c_data, color="tab:orange", label="measured")
ax[0].set(xlabel="Time (min)", ylabel="C (mol/L)", title="Kinetics fit"); ax[0].legend()
ax[1].axhline(0, color="gray", lw=1); ax[1].scatter(t_data, fit["resid"])
ax[1].set(xlabel="Time (min)", ylabel="Residual (mol/L)", title="Residuals")
save(fig, "01_kinetics_fit")

# %% [markdown]
# ## 3. Neural-network surrogate with sensor noise and a linear baseline
# The network learns pipe pressure drop from flow, diameter and length using noisy targets.
# It is scored against the clean physics on cases it never saw, then compared with linear regression.

# %%
def sample_cases(n, rough=0.045, rho=998.0, mu=1.0):
    out = []
    while len(out) < n:
        q = np.exp(rng.uniform(np.log(5), np.log(150)))
        d = np.exp(rng.uniform(np.log(25), np.log(250)))
        l = np.exp(rng.uniform(np.log(10), np.log(1000)))
        v = q / 3600 / (np.pi * (d / 1000) ** 2 / 4)
        if 0.1 <= v <= 6:
            out.append((q, d, l))
    X = np.array(out)
    return X, pipe_calc(X[:, 0], X[:, 1], X[:, 2], rough, rho, mu)["dp"]

NOISE = 0.03
X, dp_clean = sample_cases(1500)
dp_noisy = np.maximum(dp_clean * (1 + NOISE * rng.standard_normal(len(dp_clean))), dp_clean * 0.05)
ntr = int(0.8 * len(X))
Xlog = np.log(X)
sx = StandardScaler().fit(Xlog[:ntr])
y_log = np.log(dp_noisy)
y_mu, y_sd = y_log[:ntr].mean(), y_log[:ntr].std()

nn = MLPRegressor(hidden_layer_sizes=(16, 16), activation="tanh", solver="adam",
                  learning_rate_init=0.01, batch_size=64, max_iter=600,
                  alpha=1e-5, tol=1e-6, n_iter_no_change=40, random_state=7)
nn.fit(sx.transform(Xlog[:ntr]), (y_log[:ntr] - y_mu) / y_sd)
pred_nn = np.exp(nn.predict(sx.transform(Xlog[ntr:])) * y_sd + y_mu)

lin = LinearRegression().fit(X[:ntr], dp_noisy[:ntr])
pred_lin = lin.predict(X[ntr:])
truth = dp_clean[ntr:]

def mape(p, t): return np.mean(np.abs(p - t) / t) * 100
print(f"Neural net : R2 (clean) = {r2_score(truth, pred_nn):.4f}, MAPE = {mape(pred_nn, truth):.2f} %")
print(f"Linear base: R2 (clean) = {r2_score(truth, pred_lin):.4f}, MAPE = {mape(pred_lin, truth):.1f} %")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(nn.loss_curve_); ax[0].set(xlabel="Epoch", ylabel="Training loss", yscale="log", title="Training loss")
ax[1].scatter(np.log10(truth), np.log10(pred_nn), s=6, alpha=.6, label="neural net")
lim = [np.log10(truth).min(), np.log10(truth).max()]
ax[1].plot(lim, lim, "k--", lw=1)
ax[1].set(xlabel="Actual (clean) log10 dP (kPa)", ylabel="Predicted log10 dP (kPa)", title="Unseen cases"); ax[1].legend()
save(fig, "02_neural_surrogate")

def surrogate(q, d, l):
    z = sx.transform(np.log(np.array([[q, d, l]], float)))
    return float(np.exp(nn.predict(z)[0] * y_sd + y_mu))

# %% [markdown]
# ## 4. Monte Carlo uncertainty analysis

# %%
nominal = dict(q=30, d=80, l=120, rough=0.045, rho=998, mu=1.0)
sigma = dict(q=.05, d=.02, l=.01, rough=.20, rho=.01, mu=.05)
N = 5000
draw = {k: np.maximum(v * (1 + sigma[k] * rng.standard_normal(N)), 0 if k == "rough" else v * .05) for k, v in nominal.items()}
dp_mc = pipe_calc(draw["q"], draw["d"], draw["l"], draw["rough"], draw["rho"], draw["mu"])["dp"]
p5, p95 = np.percentile(dp_mc, [5, 95])
print(f"Mean dP = {dp_mc.mean():.2f} kPa, std = {dp_mc.std(ddof=1):.2f}, 5th-95th percentile = {p5:.1f} to {p95:.1f} kPa")
corr = {k: np.corrcoef(draw[k], dp_mc)[0, 1] for k in nominal}
print("Correlation with dP:", {k: round(v, 2) for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1]))})

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(dp_mc, bins=40, color="tab:cyan"); ax[0].axvline(p5, ls="--", c="tab:blue"); ax[0].axvline(p95, ls="--", c="tab:blue")
ax[0].set(xlabel="Pressure drop (kPa)", ylabel="Runs", title="Monte Carlo distribution")
names = sorted(corr, key=lambda k: abs(corr[k]))
ax[1].barh(names, [corr[k] for k in names], color=["tab:orange" if corr[k] < 0 else "tab:cyan" for k in names])
ax[1].set(xlabel="Correlation with dP", title="Input influence")
save(fig, "03_monte_carlo")

# %% [markdown]
# ## 5. Techno-economic optimisation: best pipe diameter
# Annual cost = annualised pipe cost + pumping energy cost.

# %%
def annual_cost(d_mm, q=30, l=120, a=0.35, ann=0.2, eta=0.65, price=0.10, hours=8000, **fluid):
    dp = float(pipe_calc(q, d_mm, l, **fluid)["dp"])
    power_kw = q / 3600 * dp / eta
    return ann * a * d_mm**1.4 * l + power_kw * hours * price

res = minimize_scalar(annual_cost, bounds=(20, 300), method="bounded")
STANDARD = np.array([25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300])
nearest = STANDARD[np.argmin(np.abs(STANDARD - res.x))]
print(f"Optimal diameter = {res.x:.1f} mm (nearest standard {nearest} mm), "
      f"annual cost = {annual_cost(res.x):,.0f} USD, at standard size = {annual_cost(nearest):,.0f} USD")

ds = np.linspace(30, 200, 300)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(ds, [annual_cost(d) for d in ds], label="total")
ax.axvline(res.x, color="tab:green", ls="--", label="optimum")
ax.set(xlabel="Diameter (mm)", ylabel="Annual cost (USD/yr)", ylim=(0, annual_cost(res.x) * 3), title="Optimiser"); ax.legend()
save(fig, "04_optimizer")

# %% [markdown]
# ## 6. Anomaly and fouling detection
# 200 plant readings with gradual fouling from reading 120. The surrogate predicts healthy pressure drop;
# relative residuals become z-scores using the first 60 healthy readings; |z| > 2.5 is flagged.

# %%
n, onset, fault, noise = 200, 120, 0.25, 0.03
idx = np.arange(n)
q = np.maximum(30 * (1 + 0.10 * np.sin(2 * np.pi * idx / 50) + 0.02 * rng.standard_normal(n)), 9)
healthy = pipe_calc(q, 80, 120)["dp"]
mult = np.where(idx >= onset, 1 + fault * (idx - onset) / (n - 1 - onset), 1.0)
measured = healthy * mult * (1 + noise * rng.standard_normal(n))
predicted = np.array([surrogate(qi, 80, 120) for qi in q])
rel = (measured - predicted) / predicted
z = (rel - rel[:60].mean()) / rel[:60].std(ddof=1)
flag, truth_fault = np.abs(z) > 2.5, idx >= onset
tp, fp, fn = (flag & truth_fault).sum(), (flag & ~truth_fault).sum(), (~flag & truth_fault).sum()
precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-12)
print(f"Precision = {precision:.3f}, recall = {recall:.3f}, F1 = {f1:.3f}, false alarms = {fp}")

fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(idx, z, lw=1); ax[0].scatter(idx[flag], z[flag], c="tab:orange", s=14, zorder=3)
for lim in (2.5, -2.5): ax[0].axhline(lim, color="tab:orange", ls="--")
ax[0].axvline(onset, color="tab:green", ls="--"); ax[0].set(ylabel="z-score", title="Residual control chart")
ax[1].plot(idx, measured, c="tab:orange", lw=1, label="measured"); ax[1].plot(idx, predicted, c="tab:cyan", label="predicted healthy")
ax[1].axvline(onset, color="tab:green", ls="--"); ax[1].set(xlabel="Reading", ylabel="dP (kPa)"); ax[1].legend()
save(fig, "05_anomaly_detection")
print(f"Figures saved in ./{OUT}/")
