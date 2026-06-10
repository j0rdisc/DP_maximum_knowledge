import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. MECHANISMS IMPLEMENTATION
# =============================================================================

class GoogleBoundedMeanSimulator:
    """
    Implements Google's production BoundedMean algorithm exactly as specified
    in the text: sequential composition budget splitting, global clamping,
    midpoint normalization, and a threshold floor of 1.0 for the denominator.
    """
    def __init__(self, epsilon, lower_bound, upper_bound, count_ratio=0.5):
        self.epsilon = epsilon
        self.low = lower_bound
        self.high = upper_bound
        self.count_ratio = count_ratio
        self.entries = []

    def add_entries(self, data_list):
        self.entries.extend(data_list)

    def result(self):
        data = np.array(self.entries)
        
        # 1. Split privacy budget (Sequential Composition)
        eps_count = self.epsilon * self.count_ratio
        eps_sum = self.epsilon * (1.0 - self.count_ratio)
        
        # 2. Input Clamping to public interval
        clamped_data = np.clip(data, self.low, self.high)
        
        # 3. Google's Midpoint Normalization Shift
        midpoint = (self.low + self.high) / 2.0
        shifted_data = clamped_data - midpoint
        
        # 4. Stage 1: Private Count Estimation (Sensitivity = 1)
        true_count = len(shifted_data)
        noisy_count = true_count + np.random.laplace(0.0, 1.0 / eps_count)
        
        # Enforce threshold lower bound of 1 to prevent division-by-zero or negative estimates
        if noisy_count < 1.0:
            noisy_count = 1.0
            
        # 5. Stage 2: Private Shifted Sum Estimation 
        # Sensitivity is halved by midpoint shift to: (high - low) / 2
        true_shifted_sum = np.sum(shifted_data)
        sensitivity_sum = (self.high - self.low) / 2.0
        noisy_shifted_sum = true_shifted_sum + np.random.laplace(0.0, sensitivity_sum / eps_sum)
        
        # 6. Compute Ratio and restore scale with midpoint
        private_mean = (noisy_shifted_sum / noisy_count) + midpoint
        return private_mean


class MKDPMeanMechanism:
    """
    Implements the Maximum-Knowledge Differential Privacy (MKDP) Truncated Mean.
    Bypasses budget splitting, drops empirical extremes, utilizes data-dependent
    local sensitivity, and handles range bounding via uniform mass redistribution.
    """
    def __init__(self, epsilon, lower_bound, upper_bound):
        self.epsilon = epsilon
        self.low = lower_bound
        self.high = upper_bound
        self.entries = []

    def add_entries(self, data_list):
        self.entries.extend(data_list)

    def result(self):
        # Initial cleaning clip to public interval
        s = np.sort(np.clip(np.array(self.entries), self.low, self.high))
        m = len(s)
        
        # Handle the structural boundary constraint (m > 3)
        if m <= 3:
            return np.mean(s) if m > 0 else (self.low + self.high) / 2.0
            
        s_1 = s[0]
        s_m = s[-1]
        
        # 1. Truncated Mean: drop the absolute minimum and maximum values
        truncated_subset = s[1:-1]
        truncated_mean = np.mean(truncated_subset)
        
        # 2. Local Sensitivity Calculation
        sensitivity_mk = (s_m - s_1) / (m - 3)
        if sensitivity_mk <= 0:
            sensitivity_mk = 1e-10  # Floor for degenerate datasets of identical points
            
        # 3. Add Laplace noise using the FULL undivided privacy budget
        noise = np.random.laplace(0.0, sensitivity_mk / self.epsilon)
        private_mean = truncated_mean + noise
        
        # 4. Restrict output domain to [s_1, s_m] via uniform mass redistribution
        # Lower bound: replacing r'_{m-1} with r'_1. (s[0] is r'_1, s[-2] is r'_{m-1})
        y_min = truncated_mean + (s[0] - s[-2]) / (m - 2)
        
        # Upper bound: replacing r'_2 with r'_m. (s[-1] is r'_m, s[1] is r'_2)
        y_max = truncated_mean + (s[-1] - s[1]) / (m - 2)
        
        if private_mean < y_min or private_mean > y_max:
            private_mean = np.random.uniform(y_min, y_max)
            
        return private_mean


# =============================================================================
# 2. EXPERIMENTAL TESTING REGIMES
# =============================================================================

def run_regime_1_clamping(num_trials=50, epsilon=1.0):
    """
    Regime 1: Evaluates sensitivity insulation against expanding public bounds.
    Dataset: Fixed normal distribution N(0,1), n=1000.
    Bounds: Systematic symmetric scaling from [-1,1] to [-100,100].
    """
    print("Running Regime 1: Evaluating Public Domain Clamping...")
    np.random.seed(41)
    n = 1000
    raw_data = np.random.normal(0, 1, n)
    mu_raw = np.mean(raw_data) # Ground truth comparison target
    
    intervals = np.arange(1, 101, 1)  # Represents boundary value 'b' for [-b, b]
    records = []
    
    for b in intervals:
        low, high = -float(b), float(b)
        errs_google, errs_mkdp = [], []
        
        for _ in range(num_trials):
            # Google baseline
            g_mech = GoogleBoundedMeanSimulator(epsilon, low, high)
            g_mech.add_entries(raw_data.tolist())
            errs_google.append(g_mech.result() - mu_raw)
            
            # MKDP framework
            m_mech = MKDPMeanMechanism(epsilon, low, high)
            m_mech.add_entries(raw_data.tolist())
            errs_mkdp.append(m_mech.result() - mu_raw)
            
        errs_google = np.array(errs_google)
        errs_mkdp = np.array(errs_mkdp)
        
        records.append({
            "Bound_Scale": b,
            "Google_MAE": np.mean(np.abs(errs_google)),
            "Google_MSE": np.mean(errs_google ** 2),
            "MKDP_MAE": np.mean(np.abs(errs_mkdp)),
            "MKDP_MSE": np.mean(errs_mkdp ** 2)
        })
    return pd.DataFrame(records)


def run_regime_2_sample_size(num_trials=50, epsilon=1.0):
    """
    Regime 2: Evaluates subpopulation sample size stability.
    Dataset: Fixed bounds [-5,5].
    Sample Size: Dynamic scaling from m=3 to m=100 in steps of 1.
    """
    print("Running Regime 2: Evaluating Subpopulation Sample Size...")
    np.random.seed(42)
    low, high = -3.0, 3.0
    sizes = np.arange(4, 101, 1)
    records = []
    
    for m in sizes:
        errs_google, errs_mkdp = [], []
        
        for _ in range(num_trials):
            # Generate temporary data cluster of exact size m inside the bounds
            raw_data = np.random.normal(0, 1, m)
            mu_raw = np.mean(raw_data)
            
            # Google baseline
            g_mech = GoogleBoundedMeanSimulator(epsilon, low, high)
            g_mech.add_entries(raw_data.tolist())
            errs_google.append(g_mech.result() - mu_raw)
            
            # MKDP framework
            m_mech = MKDPMeanMechanism(epsilon, low, high)
            m_mech.add_entries(raw_data.tolist())
            errs_mkdp.append(m_mech.result() - mu_raw)
            
        errs_google = np.array(errs_google)
        errs_mkdp = np.array(errs_mkdp)
        
        records.append({
            "Sample_Size": m,
            "Google_MAE": np.mean(np.abs(errs_google)),
            "Google_MSE": np.mean(errs_google ** 2),
            "MKDP_MAE": np.mean(np.abs(errs_mkdp)),
            "MKDP_MSE": np.mean(errs_mkdp ** 2)
        })
    return pd.DataFrame(records)


def run_regime_3_budget_split(num_trials=50):
    """
    Regime 3: Evaluates efficiency loss of privacy budget splitting.
    Dataset: Fixed large dataset n=1000 from N(0,1).
    Bounds: Fixed standard symmetric tight range [-5,5].
    Epsilon spectrum: Sweeping from 0.01 up to 10.0.
    """
    print("Running Regime 3: Evaluating Privacy Budget Splitting...")
    np.random.seed(42)
    n = 1000
    low, high = -3.0, 3.0
    raw_data = np.random.normal(0, 1, n)
    mu_raw = np.mean(raw_data)
    
    epsilons = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    records = []
    
    for eps in epsilons:
        errs_google, errs_mkdp = [], []
        
        for _ in range(num_trials):
            # Google baseline
            g_mech = GoogleBoundedMeanSimulator(eps, low, high)
            g_mech.add_entries(raw_data.tolist())
            errs_google.append(g_mech.result() - mu_raw)
            
            # MKDP framework
            m_mech = MKDPMeanMechanism(eps, low, high)
            m_mech.add_entries(raw_data.tolist())
            errs_mkdp.append(m_mech.result() - mu_raw)
            
        errs_google = np.array(errs_google)
        errs_mkdp = np.array(errs_mkdp)
        
        records.append({
            "Epsilon": eps,
            "Google_MAE": np.mean(np.abs(errs_google)),
            "Google_MSE": np.mean(errs_google ** 2),
            "MKDP_MAE": np.mean(np.abs(errs_mkdp)),
            "MKDP_MSE": np.mean(errs_mkdp ** 2)
        })
    return pd.DataFrame(records)


# =============================================================================
# 3. HIGH-QUALITY PLOTTING IMPLEMENTATION
# =============================================================================

def plot_regime_results(df, x_col, title_prefix, file_name, use_log_x=False):
    """Generates two-panel utility comparison figures for MAE and MSE."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = [("MAE", "Google_MAE", "MKDP_MAE"), ("MSE", "Google_MSE", "MKDP_MSE")]
    
    for ax, (label, g_col, m_col) in zip(axes, metrics):
        ax.grid(True, which="both", ls=":", alpha=0.5)
        
        ax.plot(df[x_col], df[g_col], linestyle="--", marker="o", markersize=4, color="#d95f02", linewidth=2.0, label="Google BoundedMean")
        ax.plot(df[x_col], df[m_col], linestyle="-", marker="s", markersize=4, color="#1b9e77", linewidth=2.0, label="MKDP (Proposed)")
        
        ax.set_title(f"{label} Performance", fontsize=12)
        ax.set_xlabel(x_col.replace("_", " "), fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_yscale("log")
        if use_log_x:
            ax.set_xscale("log")
        ax.legend(frameon=True, fontsize=10)
        
    plt.suptitle(title_prefix, fontsize=14, y=0.98)
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close()


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Run Regime 1
    df1 = run_regime_1_clamping()
    df1.to_csv("mean_regime1_clamping.csv", index=False)
    plot_regime_results(df1, "Bound_Scale", "Mean Utility Profile vs. Expanding Public Domain Clamping", "mean_regime1_clamping.png")
    
    # Run Regime 2
    df2 = run_regime_2_sample_size()
    df2.to_csv("mean_regime2_samplesize.csv", index=False)
    plot_regime_results(df2, "Sample_Size", "Mean Utility Profile vs. Subpopulation Sample Size", "mean_regime2_samplesize.png")
    
    # Run Regime 3
    df3 = run_regime_3_budget_split()
    df3.to_csv("mean_regime3_budgetsplit.csv", index=False)
    plot_regime_results(df3, "Epsilon", "Mean Utility Profile vs. Total Privacy Budget (Epsilon)", "mean_regime3_budgetsplit.png", use_log_x=True)
    
    print("\nAll experiments complete. Output CSV logs and plot figures generated successfully.")