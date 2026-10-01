import numpy as np
import matplotlib.pyplot as plt

# --- Pure NumPy Laplace Functions ---
def laplace_pdf(x, loc, scale):
    """Laplace Probability Density Function."""
    return (1.0 / (2.0 * scale)) * np.exp(-np.abs(x - loc) / scale)

def laplace_cdf(x, loc, scale):
    """Laplace Cumulative Distribution Function."""
    if x < loc:
        return 0.5 * np.exp((x - loc) / scale)
    else:
        return 1.0 - 0.5 * np.exp(-(x - loc) / scale)

# --- MKDP Density & Sampling Functions ---
def mkdp_pdf(x, y_true, delta, eps, y_min, y_max):
    """Computes p(x) for MKDP with uniform reallocation over [y_min, y_max]."""
    if x < y_min or x > y_max:
        return 0.0
    
    scale = delta / eps
    P_in = laplace_cdf(y_max, y_true, scale) - laplace_cdf(y_min, y_true, scale)
    P_out = 1.0 - P_in
    
    density = laplace_pdf(x, y_true, scale) + (P_out / (y_max - y_min))
    return density

def sample_mkdp(y_true, delta, eps, y_min, y_max, n_samples=50000):
    """Draws n_samples from the MKDP mechanism via uniform reallocation."""
    scale = delta / eps
    samples = []
    
    while len(samples) < n_samples:
        raw = np.random.laplace(loc=y_true, scale=scale, size=n_samples)
        for val in raw:
            if y_min <= val <= y_max:
                samples.append(val)
            else:
                samples.append(np.random.uniform(y_min, y_max))
            if len(samples) == n_samples:
                break
    return np.array(samples)

# --- Simulation Setup ---
np.random.seed(42)
epsilon = 1.0
n_P = 100

# 1. True Dataset D (Sorted)
D = np.sort(np.random.normal(loc=0, scale=1, size=n_P))

# 2. Compute TMean(D) excluding initial (D[0]) and final (D[-1]) elements
y_D = np.mean(D[1:-1])

# 3. Contextual Sensitivity and Eq. (2) Feasible Interval U derived strictly from D
delta = (D[-1] - D[0]) / (n_P - 3)
y_min = y_D + (D[0] - D[-2]) / (n_P - 2)
y_max = y_D + (D[-1] - D[1]) / (n_P - 2)

# 4. Worst-case Neighbor D': Swap r'_2 (D[1]) with r'_{n_P} (D[-1])
D_prime = D.copy()
D_prime[1] = D[-1]
D_prime = np.sort(D_prime)
y_Dp = np.mean(D_prime[1:-1])

# --- Generate Output Samples from MKDP(D) ---
N = 50000
outputs = sample_mkdp(y_D, delta, epsilon, y_min, y_max, n_samples=N)

# --- Compute Empirical Log-Likelihood Ratio L_{D, D'}(x) ---
privacy_losses = []
for x in outputs:
    p_D = mkdp_pdf(x, y_D, delta, epsilon, y_min, y_max)
    p_Dp = mkdp_pdf(x, y_Dp, delta, epsilon, y_min, y_max)
    
    if p_Dp > 0 and p_D > 0:
        loss = np.log(p_D / p_Dp)
        privacy_losses.append(loss)

# --- Plot Empirical Privacy Loss Distribution ---
plt.figure(figsize=(7, 4))
plt.hist(privacy_losses, bins=100, density=True, alpha=0.7, color='navy', label='Empirical Loss')
plt.axvline(x=epsilon, color='red', linestyle='--', label=r'$+\epsilon$ bound')
plt.axvline(x=-epsilon, color='red', linestyle='--', label=r'$-\epsilon$ bound')

plt.xlabel(r"Privacy Loss $L_{D, D'}(x) = \ln(p_D(x) / p_{D'}(x))$")
plt.ylabel('Density')
plt.title(r'Empirical Privacy Loss Distribution ($\epsilon = 1.0$)')
plt.legend()
plt.tight_layout()
plt.show()
