import pandas as pd
import matplotlib.pyplot as plt
import os

def export_training_plots(file_path1, file_path2, labels, target_columns=None, output_folder="comparison_plots"):
    try:
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Filter both for epochs 1-20
    df1 = df1[(df1['epoch'] >= 1) & (df1['epoch'] <= 20)]
    df2 = df2[(df2['epoch'] >= 1) & (df2['epoch'] <= 20)]

    if target_columns is None:
        target_columns = {col: 'Value' for col in df1.columns if col != 'epoch'}

    for metric, y_label in target_columns.items():
        plt.figure(figsize=(10, 6))
        
        # Plot Model 1
        plt.plot(df1['epoch'], df1[metric], color='tab:blue', marker='o', linewidth=2, label=labels[0])
        # Plot Model 2
        plt.plot(df2['epoch'], df2[metric], color='tab:orange', marker='s', linewidth=2, label=labels[1])
        
        plt.title(f'Training Comparison: {metric.replace("_", " ").title()}', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel(y_label)
        plt.xticks(range(1, 21))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(output_folder, f"comparison_{metric}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    my_metrics = {
        'reached_goal': 'Success Count',
        'mean_unique_collisions': 'Average Collisions',
        'mean_alignment': 'Alignment Score',
        'mean_path_efficiency': 'Efficiency Ratio',
        'mean_throttle': 'Throttle Input'
    }
    
    # Pass both files and their corresponding labels
    export_training_plots(
        'training_metrics_JEPA.csv', 
        'training_metrics_LSTM.csv', 
        labels=['JEPA Architecture', 'Recurrent Architecture'],
        target_columns=my_metrics
    )