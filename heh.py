import re
import csv


def round_numbers_in_line(line):
    """
    Rounds all numbers in a line to 2 decimal places.
    """
    return re.sub(r'\b\d+(\.\d+)?\b', lambda match: f"{float(match.group()):.3f}", line)


def process_file(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    processed_lines = []
    previous_line = None
    round_id = 0
    for line in lines:
        # Remove duplicate consecutive lines
        if line == previous_line:
            continue
        previous_line = line

        # Skip lines starting with 'round'
        if line.strip().lower().startswith('round'):
            continue

        # Remove specific words and clean the line
        for word in ["Evaluation: Loss:", "Accuracy:", "F1 Score:", "Kappa:", "ROC AUC:"]:
            line = line.replace(word, "").strip()

        if line:  # Only add non-empty lines
            round_id += 1
            processed_lines.append(str(round_id)+", "+line)

    # Prepare data for CSV
    csv_data = [["Round", "Loss", "Accuracy", "F1_score", "Kappa", "ROC_auc"]]
    for line in processed_lines:
        csv_data.append(line.split(","))

    # Write to output CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)


# Example usage
files = ["rapport_FedAvg_Iid_attackers_0", "rapport_FedAvg_Non-Iid_attackers_0",
         "rapport_FedAvg_Iid_attackers_1", "rapport_FedAvg_Non-Iid_attackers_1",
         "rapport_FedAvg_Iid_attackers_2", "rapport_FedAvg_Non-Iid_attackers_2",
         "rapport_FedProx_Iid_attackers_0", "rapport_FedProx_Non-Iid_attackers_0",
         "rapport_FedProx_Iid_attackers_1", "rapport_FedProx_Non-Iid_attackers_1",
         "rapport_FedProx_Iid_attackers_2", "rapport_FedProx_Non-Iid_attackers_2"]

for file in files:
    input_filename = f"log/{file}.txt"
    output_filename = f"log/clean/{file}.csv"
    process_file(input_filename, output_filename)
