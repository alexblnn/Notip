# %%

import matplotlib.pyplot as plt
import numpy as np

# Define your parameters
n_tests = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
nb_methods = 5
labels = ['Permutation - Separate datasets',
'Permutation - two rounds - single dataset',
'Permutation - single round - single dataset',
'Bootstrap  - single round - single dataset',
'Sample Splitting']
alpha = 0.1
N = 100  # Sample size

pi0 = 1
FWHM = 8

# Load your data for bounds (shape: len(fwhms), nb_methods)
bounds = np.load(f"../figures/jers_n_sam_pi0{pi0}_fwhm{FWHM}.npy")

#remove first two values
n_tests = n_tests[2:]
bounds = bounds[2:]

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
    plt.plot(n_tests, bounds[:, i], label=labels[i])
    
    # Fill the interval around the curve using the binomial standard deviation
    # plt.fill_between(fwhms, lower_bound, upper_bound, alpha=0.2)

plt.xscale('log')
# Add a horizontal red line at alpha level
#plt.axhline(y=alpha, color='red', linestyle='--', label=f'α = {alpha}')
plt.axhline(y=alpha, color='red', linestyle='--', label=f'α = {alpha}')

# Labeling the plot
plt.xlabel('n')
plt.ylabel('Joint Error Rate (JER)')
plt.title(fr"$\pi_{0} = {pi0}$, FWHM = {FWHM}")
plt.legend(prop={'size': 8})

# Show and save the plot

plt.savefig(f"fig_sam_jer_n_tests_pi0{pi0}_fwhm{FWHM}.pdf", bbox_inches='tight')
plt.show()



# %%
bounds = np.load(f"../figures/powers_n_sam_pi0{pi0}_fwhm{FWHM}.npy")
bounds = bounds[2:]

plt.figure(figsize=(6, 4))
for i in range(nb_methods):
    # Plot the main curve
    plt.plot(n_tests, bounds[:, i], label=labels[i])


plt.xscale('log')
plt.xlabel('n')
plt.ylabel('Power')
plt.title(fr"$\pi_{0} = {pi0}$, FWHM = {FWHM}")
plt.legend(prop={'size': 8})

plt.savefig(f"fig_sam_power_n_tests_pi0{pi0}_fwhm{FWHM}.pdf", bbox_inches='tight')
plt.show()
# %%
