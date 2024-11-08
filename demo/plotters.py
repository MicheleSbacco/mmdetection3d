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



def plot_losses_metrics(epochs, training_losses, validation_losses, metric_ap40, metric_ap40_reduced):
    # Plot the training and validation losses
    plt.figure(figsize=(15, 10))
    plt.plot(epochs, training_losses, 'b-', marker='o', label='Training Loss')  # Blue line for training
    plt.plot(epochs, validation_losses, 'r-', marker='o', label='Validation Loss')  # Red line for validation
    plt.plot(epochs, metric_ap40, 'k-', marker='o', label='Metric: 3D AP40')  # Black line for ap40
    plt.plot(epochs, metric_ap40_reduced, 'g-', marker='o', label=f'Metric: 3D AP40$_{{\\mathrm{{reduced}}}}$')  # Green line for ap40

    # Set axes to start from 0
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True)

    # Fontsizes
    fontsize_titles = 16
    fontsize_annotations = 8
    superposition_upper_limit = 3
    superposition_lower_limit = 1/3

    # Add labels and title
    plt.xlabel('Epochs', fontsize = fontsize_titles)
    plt.ylabel('Metrics', fontsize = fontsize_titles)
    plt.title('Training vs Validation Loss', fontsize = fontsize_titles)

    # Add annotations for each point (training)
    for i, loss in enumerate(training_losses):
        plt.annotate(f'{loss:.2f}', (epochs[i], training_losses[i]), textcoords="offset points", xytext=(-15,-15), ha='center', fontsize=fontsize_annotations, color='blue')

    # Add annotations for each point (validation)
    for i, loss in enumerate(validation_losses):
        plt.annotate(f'{loss:.2f}', (epochs[i], validation_losses[i]), textcoords="offset points", xytext=(-15, 8), ha='center', fontsize=fontsize_annotations, color='red')

    # Add annotations for each point (ap40 metric)
    for i, ap40 in enumerate(metric_ap40):
        if (ap40/training_losses[i]>superposition_upper_limit or ap40/training_losses[i]<superposition_lower_limit) and (ap40/validation_losses[i]>superposition_upper_limit or ap40/validation_losses[i]<superposition_lower_limit):
            plt.annotate(f'{ap40:.2f}', (epochs[i], metric_ap40[i]), textcoords="offset points", xytext=(0,7), ha='center', fontsize=fontsize_annotations, color='black')
        else:
            plt.annotate(f'', (epochs[i], metric_ap40[i]))

    # Add annotations for each point (ap40_reduced metric)
    for i, ap40_r in enumerate(metric_ap40_reduced):
        if (ap40_r/training_losses[i]>superposition_upper_limit or ap40_r/training_losses[i]<superposition_lower_limit) and (ap40_r/validation_losses[i]>superposition_upper_limit or ap40_r/validation_losses[i]<superposition_lower_limit):
            plt.annotate(f'{ap40_r:.2f}', (epochs[i], metric_ap40_reduced[i]), textcoords="offset points", xytext=(0,7), ha='center', fontsize=fontsize_annotations, color='green')
        else:
            plt.annotate(f'', (epochs[i], metric_ap40[i]))

    # Show the legend
    plt.legend(fontsize = 14)

    # Display the plot
    plt.show()



def plot_precision_recall_curve(epoch, precisions, recalls, precisions_reduced, recalls_reduced, iou_thr_list):
    # Plotting the precision-recall curve
    plt.figure(figsize=(12, 12))
    plt.plot(recalls, precisions, marker='o', linestyle='-', color='k', label=f"P-R curve for AP40 at epoch {epoch}")
    plt.plot(recalls_reduced, precisions_reduced, marker='o', linestyle='-', color='g', label=f"P-R curve for AP40$_{{\\mathrm{{reduced}}}}$ at epoch {epoch}")
    
    # Annotate the IoU threshold values
    for i, thr in enumerate(iou_thr_list):
        # AP40_reduced
        if i == 0 or (recalls_reduced[i] != recalls_reduced[i-1]) or (precisions_reduced[i] != precisions_reduced[i-1]):
            plt.annotate(f'@{thr:.3f}', (recalls_reduced[i], precisions_reduced[i]), textcoords="offset points", xytext=(15,7), ha='center', fontsize=10, color='blue')

    # Set x and y axis limits dynamically
    max_recall = max(max(recalls), max(recalls_reduced))
    max_precision = max(max(precisions), max(precisions_reduced))
    limit_overall = max(max_precision, max_recall)*1.1
    plt.xlim([0, limit_overall])
    plt.ylim([0, limit_overall])
    
    # Adding labels, title, and grid
    plt.xlabel('Recall', fontsize = 16)
    plt.ylabel('Precision', fontsize = 16)
    plt.title('Precision-Recall Curve', fontsize = 16)
    plt.legend(fontsize = 14)
    plt.grid(True)
    
    # Display the plot
    plt.show()
