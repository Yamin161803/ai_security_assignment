import pandas as pd
import matplotlib.pyplot as plt


def save_plot_csv_data(csv_file, output_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Set the 'Round' column as the index
    data.set_index('Round', inplace=True)

    # Plot each column
    plt.figure(figsize=(10, 6))
    for column in data.columns:
        plt.plot(data.index, data[column], marker='o', label=column)

    # Add plot details
    plt.title(csv_file.split("/")[2][:-4], fontsize=16)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.legend(title="Metrics", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.ylim(0.0, 2.5)

    # Save the plot to a file
    plt.savefig(output_file, dpi=300)
    plt.close()  # Close the plot to free memory


# Example usage
# Replace with your actual file name
# Example usage
files = ["rapport_FedAvg_Iid_attackers_0", "rapport_FedAvg_Non-Iid_attackers_0",
         "rapport_FedAvg_Iid_attackers_1", "rapport_FedAvg_Non-Iid_attackers_1",
         "rapport_FedAvg_Iid_attackers_2", "rapport_FedAvg_Non-Iid_attackers_2",
         "rapport_FedProx_Iid_attackers_0", "rapport_FedProx_Non-Iid_attackers_0",
         "rapport_FedProx_Iid_attackers_1", "rapport_FedProx_Non-Iid_attackers_1",
         "rapport_FedProx_Iid_attackers_2", "rapport_FedProx_Non-Iid_attackers_2"]

for file in files:
    input_filename = f"log/clean/{file}.csv"
    output_filename = f"log/plot/{file}.png"
    save_plot_csv_data(input_filename, output_filename)
