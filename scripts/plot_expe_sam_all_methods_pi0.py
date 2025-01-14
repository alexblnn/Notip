# %%
import matplotlib.pyplot as plt
import numpy as np

# Define your parameters
pi0s = [0.5, 0.6, 0.8, 0.9, 1]
nb_methods = 4
labels = ['Vanilla', 'Two rounds', 'One round', 'Bootstrap']
alpha = 0.1
N = 1000  # Sample size

# Load your data for bounds (shape: len(fwhms), nb_methods)
bounds = np.load("../figures/jers_pi0_sam.npy")
for i in range(len(pi0s)):
    bounds[i] = bounds[i]/(pi0s[i] * alpha)

# Calculate the binomial standard deviation for each method (same across FWHMs as per binomial law)
p = 1 - alpha
binomial_std = np.sqrt(N * p * (1 - p)) / N  # Normalize by N to keep the scale as a proportion

# Plotting the JER curves
plt.figure(figsize=(6, 4))

# Plot each method's JER curve with a shaded interval using binomial standard deviation
for i in range(nb_methods):
    # Calculate upper and lower bounds for the shaded interval using binomial standard deviation
    lower_bound = bounds[:, i] - binomial_std
    upper_bound = bounds[:, i] + binomial_std
    
    # Plot the main curve
    plt.plot(pi0s, bounds[:, i], label=labels[i])
    
    # Fill the interval around the curve using the binomial standard deviation
    # plt.fill_between(fwhms, lower_bound, upper_bound, alpha=0.2)

# Add a horizontal red line at alpha level
#plt.axhline(y=alpha, color='red', linestyle='--', label=f'α = {alpha}')

# Labeling the plot
plt.xlabel(r'$\pi_0$')
plt.ylabel(r'$JER/\pi_0 \alpha$')
plt.legend()

# Show and save the plot

plt.savefig("fig_sam_jer_pi0s.pdf")
plt.show()



# %%
