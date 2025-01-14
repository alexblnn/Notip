# %%
import matplotlib.pyplot as plt
import numpy as np

# Define your parameters
fwhms = [0, 2, 4, 6, 8]
nb_methods = 4
labels = ['Vanilla', 'Two rounds', 'One round', 'Bootstrap']
alpha = 0.05
N = 100  # Sample size

# Load your data for bounds (shape: len(fwhms), nb_methods)
bounds = np.load("jers_fwhm_1st_try.npy")

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
    plt.plot(fwhms, bounds[:, i], label=labels[i])
    
    # Fill the interval around the curve using the binomial standard deviation
    # plt.fill_between(fwhms, lower_bound, upper_bound, alpha=0.2)

# Add a horizontal red line at alpha level
plt.axhline(y=alpha, color='red', linestyle='--', label=f'α = {alpha}')

# Labeling the plot
plt.xlabel('FWHM (mm)')
plt.ylabel('Joint Error Rate (JER)')
plt.legend()

# Show and save the plot

plt.savefig("fig_sam_jer_fwhm.pdf")
plt.show()



# %%
