import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Read the CSV file into a DataFrame
experiments = pd.read_csv('./outputs/experiments-glue.csv')

# Create the plot
plt.figure(figsize=(6, 4))

# Plot data for each method
for method in experiments['method'].unique():
    runs = experiments[experiments['method'] == method]
    if not runs.empty:
        stats = runs.groupby('trainable_params')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
        stats['sem'] = stats['std'] / np.sqrt(stats['count'])
        
        plt.plot(
            stats['trainable_params'], 
            stats['mean'], 
            '-', linewidth=2,
            label="Adapters" if method == "adapters" else "LoRA" if method == "lora" else "Fine-tune top k layers"
        )
        
        plt.fill_between(
            stats['trainable_params'],
            stats['mean'] - 3 * stats['std'],
            stats['mean'] + 3 * stats['std'],
            alpha=0.2
        )

plt.xscale('log')
plt.xlim([0.3e5, 0.2e9])
plt.ylim([0.575, 0.875])
plt.xlabel('Number of Trainable Parameters')
plt.ylabel('Validation Accuracy')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("MultiNLI-matched")
plt.tight_layout()
plt.savefig('./outputs/mnli_experiments.png', dpi=300, bbox_inches='tight')
plt.show()