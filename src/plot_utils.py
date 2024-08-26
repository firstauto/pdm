import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# plt.rcParams.update({
#     'figure.figsize': (6, 5),
#     'font.family': 'Times New Roman',
#     'font.size': 16,
#     'lines.linewidth': 1.5,
#     'lines.markersize': 8,
#     'legend.fontsize': 14,
#     'axes.titlesize': 14,
#     'axes.labelsize': 15,
#     'axes.labelweight': 'bold',
#     'xtick.labelsize': 15,
#     'ytick.labelsize': 15,
#     'figure.dpi': 600,
# })

def weights_heatmap(tensor, name):
    """
    Generate heatmaps for tensor weights and save them as images.

    Parameters:
    tensor (torch.Tensor or list of torch.Tensor): The tensor containing weights.
    name (str): The base name for the saved heatmap images.
    """
    # Directory to save the heatmaps
    save_dir = 'G:/BN_research/dataFromPeng/code/MPHT/visuals/weights/'
    os.makedirs(save_dir, exist_ok=True)

    def save_heatmap(data, filename):
        """Helper function to save a single heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, cmap='viridis')
        # plt.xticks(np.arange(0, data.shape[1], step=8), fontsize=8, rotation=90)
        # plt.yticks(np.arange(0, data.shape[0], step=8), fontsize=8, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=600)
        plt.close()

    if isinstance(tensor, list):
        # Handle list of tensors (e.g., multi-head attention weights)
        for i, head_tensor in enumerate(tensor):
            head_data = head_tensor[0].cpu().detach().numpy()
            save_heatmap(head_data, f"head_{i}_{name}.png")
    else:
        if tensor.dim() == 3:
            # Handle 3D tensor (e.g., batch of weight matrices)
            num_samples = min(4, tensor.size(0))  # Limit to 4 samples
            sample_indices = np.random.choice(tensor.size(0), num_samples, replace=False)
            for i in sample_indices:
                sample_data = tensor[i].cpu().detach().numpy()
                save_heatmap(sample_data, f"num_{i}_{name}.png")
        elif tensor.dim() == 2:
            # Handle 2D tensor (e.g., single weight matrix)
            data = tensor.cpu().detach().numpy()
            save_heatmap(data, f"2D_{name}.png")
        else:
            raise ValueError("Tensor must be either 2D or 3D")

def plot_output(targets, outputs, plot_name):

    zipped = sorted(zip(targets, outputs))
    grouped = {}
    for key, value in zipped:
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(value)

    # Plotting
    means = {key: np.mean(values) for key, values in grouped.items()}
    min_values = {key: min(values) for key, values in grouped.items()}
    max_values = {key: max(values) for key, values in grouped.items()}

    keys = sorted(means.keys())
    means_plot = [means[key] for key in keys]
    min_plot = [min_values[key] for key in keys]
    max_plot = [max_values[key] for key in keys]

    # Create the plot
    plt.figure(figsize=(6, 5))
    plt.plot(keys, keys, 'r', label='Actual Values')  # Red line for true values (diagonal)
    plt.plot(keys, means_plot, color='blue', label='Predicted Values')
    plt.fill_between(keys, min_plot, max_plot, color='royalblue', alpha=0.5, label='Cofidence Interval')

    targets, outputs = zip(*zipped)
    targets, outputs  = np.array(targets), np.array(outputs)
    errors = targets - outputs
    # Error metrics
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(1 + np.abs((errors) /(targets + 1)) - 1) * 100
    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    plt.title(f'RMSE: {rmse:.4f} || MAE: {mae:.4f}')
    plt.xlabel('Experimental Fatigue life / cycles', fontweight='bold')
    plt.ylabel('Predicted Fatigue life / cycles', fontweight='bold')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    # Save the plot
    dir = 'G:/turbine_ad/visuals/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{dir}{plot_name}_normal.png', dpi=600)
    plt.close()
    return targets, outputs


def plot_output_recon(targets, outputs, num_sample, system, features, time_stamp, plot_name):

    # Ensure time_stamp is a DataFrame and contains datetime data
    stamp = pd.DataFrame(time_stamp, columns=['Timestamp'])
    stamp['Timestamp'] = pd.to_datetime(stamp['Timestamp'])  # Convert to datetime if not already
    stamp['date'] = stamp['Timestamp'].dt.date
    stamp['time'] = stamp['Timestamp'].dt.time
    stamp.reset_index(drop=True, inplace=True)

    num_features = targets.shape[-1]
    num_rows = (num_features + 1) // 2  # Calculate the number of rows needed for 2 columns

    plt.figure(figsize=(18, num_rows * 3))  # Adjust the figure height based on the number of rows
    for i in range(num_features):
        ax = plt.subplot(num_rows, 2, i + 1)  # Arrange subplots in num_rows x 2 grid
        plt.plot(targets[num_sample, :, i].reshape(-1,), color='red', label='Actual')  # Red line for true values
        plt.plot(outputs[num_sample, :, i].reshape(-1,), color='blue', label='Reconstructed')

        errors = targets[num_sample, :, i].reshape(-1,) - outputs[num_sample, :, i].reshape(-1,)
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))

        plt.title(f'RMSE: {rmse:.4f} || MAE: {mae:.4f}')
        plt.ylabel(f'{features[i]}', fontweight='bold')

        # Only show xticks and xlabel for subplots in the last row
        if i < num_features - 2:
            ax.set_xticklabels([])  # Hide x-axis labels for all but the last row
            ax.set_xticks([])       # Hide x-axis ticks for all but the last row
        else:
            # Set xticks and xlabel for the subplots in the last row
            tick_interval = len(stamp) // 10 if len(stamp) > 10 else 1
            plt.xticks(np.arange(0, len(stamp), tick_interval), stamp['time'][::tick_interval], rotation=45) # type: ignore
            plt.xlabel(f"Time {stamp['date'].iloc[0]} - {stamp['date'].iloc[-1]}", fontweight='bold')

    plt.legend()
    # plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()

    # Ensure the directory exists and save the plot
    dir = f'{os.getcwd()}/visuals/{system}/'
    os.makedirs(dir, exist_ok=True)
    
    plt.savefig(f"{dir}{plot_name}_{num_sample}_normal.png", dpi=600)
    plt.close()



def anomaly_plot(targets, outputs, anomaly_idx, system, features, time_stamp, plot_name):
    # Ensure time_stamp is a DataFrame and contains datetime data
    stamp = pd.DataFrame(time_stamp, columns=['Timestamp'])
    stamp['Timestamp'] = pd.to_datetime(stamp['Timestamp'])  # Convert to datetime if not already
    stamp['date'] = stamp['Timestamp'].dt.date
    stamp['time'] = stamp['Timestamp'].dt.time
    stamp.reset_index(drop=True, inplace=True)

    num_features = targets.shape[-1]
    num_rows = (num_features + 1) // 2  # Calculate the number of rows needed for 2 columns

    plt.figure(figsize=(18, num_rows * 3))  # Adjust the figure height based on the number of rows
    for i in range(num_features):
        ax = plt.subplot(num_rows, 2, i + 1)  # Arrange subplots in num_rows x 2 grid
        plt.plot(targets[anomaly_idx, :, i].reshape(-1,), color='red', label='Actual')  # Red line for true values
        plt.plot(outputs[anomaly_idx, :, i].reshape(-1,), color='blue', label='Reconstructed')

        errors = targets[anomaly_idx, :, i].reshape(-1,) - outputs[anomaly_idx, :, i].reshape(-1,)
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))

        plt.title(f'RMSE: {rmse:.4f} || MAE: {mae:.4f}')
        plt.ylabel(f'{features[i]}', fontweight='bold')

        # Only show xticks and xlabel for subplots in the last row
        if i < num_features - 2:
            ax.set_xticklabels([])  # Hide x-axis labels for all but the last row
            ax.set_xticks([])       # Hide x-axis ticks for all but the last row
        else:
            # Set xticks and xlabel for the subplots in the last row
            tick_interval = len(stamp) // 10 if len(stamp) > 10 else 1
            plt.xticks(np.arange(0, len(stamp), tick_interval), stamp['time'][::tick_interval], rotation=45) # type: ignore
            plt.xlabel(f"Time {stamp['date'].iloc[0]} - {stamp['date'].iloc[-1]}", fontweight='bold')

    plt.legend()
    # plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()

    # Ensure the directory exists and save the plot
    dir = f'{os.getcwd()}/visuals/{system}/'
    os.makedirs(dir, exist_ok=True)
    
    plt.savefig(f"{dir}{plot_name}_{stamp['date'].iloc[0]}_{anomaly_idx}.png", dpi=600)
    plt.close()


