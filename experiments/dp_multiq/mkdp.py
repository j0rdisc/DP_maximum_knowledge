import numpy as np

def mkdp(sorted_data, data_low, data_high, qs, eps):
  """Returns eps-DP quantile estimates for qs under the maximum knowledge assumption.
  
  Args:
    sorted_data: Array of data points sorted in increasing order.
    data_low: Lower limit for any differentially private quantile output value.
    data_high: Upper limit for any differentially private quantile output value.
    qs: Increasing array of quantiles in [0,1].
    eps: Privacy parameter epsilon.
  """
  clamped_data = np.clip(sorted_data, data_low, data_high)
  n = len(clamped_data)
  
  # Ensure qs is a flattened iterable array sequence
  qs = np.atleast_1d(qs)
  m = len(qs)
  outputs = np.empty(m)
  
  # Split the privacy budget across the requested quantiles to preserve overall eps-DP
  divided_eps = eps / m if m > 0 else eps
  
  # Pad the data array with boundary limits to safely resolve index boundaries (j-1 and j+1)
  padded_data = np.block([data_low, clamped_data, data_high])
  
  for idx, q in enumerate(qs):
    # CRITICAL FIX: Explicitly cast q to a native float primitive.
    # This keeps array pollution from breaking native methods downstream.
    q_scalar = float(q.item() if hasattr(q, "item") else q)
    
    # Determine the target index matching the lower-quantile rule
    j = int(np.floor((n - 1) * q_scalar))
    
    # In padded_data, the original element D[j] is shifted to index j + 1
    y_min = padded_data[j]       # Equivalent to D[j-1] or data_low
    y_max = padded_data[j + 2]   # Equivalent to D[j+1] or data_high
    
    # Adapted local sensitivity over the feasible space M_D
    sensitivity = y_max - y_min
    true_val = clamped_data[j]
    
    # Handle homogeneous neighborhoods (sensitivity of 0 yields a 0-DP exact release)
    if sensitivity <= 0.0:
      outputs[idx] = true_val
      continue
      
    # Add Laplace noise calibrated to the adapted sensitivity
    scale = sensitivity / divided_eps
    raw_noise = np.random.laplace(0.0, scale)
    candidate = true_val + raw_noise
    
    # Proposition 3: Uniform Max-Entropy Reallocation for out-of-bounds mass
    if candidate < y_min or candidate > y_max:
      outputs[idx] = np.random.uniform(y_min, y_max)
    else:
      outputs[idx] = candidate

  return np.sort(outputs)