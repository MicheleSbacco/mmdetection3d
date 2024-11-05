import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

def freq_plot(values, title = "Frequency Plot"):
    # Calculating the average value of the given list
    average_value = sum(values) / len(values)

    # Plotting the frequency plot with a vertical red line for the average value
    plt.figure(figsize=(15, 10))
    plt.hist(values, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(average_value, color='red', linestyle='dashed', linewidth=2, label=f'Average: {(average_value*1000):.2f}')
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

def freq_plot_with_gaussian(values, title="Frequency Plot with Gaussian", columns_color="blue", n_bins = 80):
    # Set everything to [ms] instead of [s]
    values = [v*1000 for v in values]

    # Calculating the average value and the sample variance of the given list
    average_value = sum(values) / len(values)
    sample_variance = sum((x - average_value) ** 2 for x in values) / (len(values) - 1)
    sample_std_dev = np.sqrt(sample_variance)

    # Plotting the frequency plot with a vertical red line for the average value
    plt.figure(figsize=(10, 5))
    n_bins = n_bins
    plt.hist(values, bins=n_bins, alpha=0.7, color=columns_color, edgecolor='black', density=False)
    plt.axvline(average_value, color='red', linestyle='--', linewidth=3, label=f'$\mu={average_value:.2f}$ ms')

    # Plotting the Gaussian curve
    x = np.linspace(min(values), max(values), 1000)
    y = norm.pdf(x, average_value, sample_std_dev) * len(values) * (max(values) - min(values)) / n_bins
    plt.plot(x, y, color='red', linestyle='-', linewidth=3, label=f'$\sigma={sample_std_dev:.2f}$ ms')

    plt.title(title, fontsize = 16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.show()

def plot_pie_chart(values_1, values_2, values_3):
    average_value1 = sum(values_1) / len(values_1)
    average_value2 = sum(values_2) / len(values_2)
    average_value3 = sum(values_3) / len(values_3)
    sizes = [average_value1, average_value2, average_value3]
    labels = ["Pre-processing", "Inference", "Post-processing"]
    colors = ['green', 'gold', 'gray']
    explosion = [.05, .05, .05]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explosion, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, 
            textprops={'fontsize': 20},
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.75})
    plt.axis('equal')
    plt.show()
