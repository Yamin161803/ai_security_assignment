"""
    Authors:     
    Philip Wollsén Ervius
    phao21@student.bth.se

    Amin Afzali
    moaf@student.bth.se

    """

import flwr as fl
from flwr.common import Context
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np


# Global variables
round_int = 0
server_address = "localhost:3000"
nr_of_rounds = 50
nr_of_clients = 5
client_id_file = "client_ids.txt"
nr_attackers = 2
file_name = "log/attcker_2.txt"

# Get datasets
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

dataset = datasets.CIFAR10(
    root='./datasets', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(
    root='./datasets', train=False, download=True, transform=transform)


def iid_partition_data(dataset, num_clients):
    partitions = random_split(
        dataset, [len(dataset) // num_clients] * num_clients)
    return [DataLoader(part, batch_size=32, shuffle=True) for part in partitions]


def dirichlet_partitioner(dataset, num_clients, alpha=0.5, num_classes=10):

    targets = np.array(dataset.targets)
    classes = np.unique(targets)

    # Initialize a list to hold client indices
    client_indices = {i: [] for i in range(num_clients)}

    # Sample Dirichlet distribution to determine the proportion of classes for each client
    dirichlet_samples = np.random.dirichlet([alpha] * num_classes, num_clients)

    # Create a dictionary to store indices for each class
    class_indices = {cls: np.where(targets == cls)[0] for cls in classes}

    # Assign data to clients based on the Dirichlet distribution
    for client_id in range(num_clients):
        for cls in classes:
            # Calculate the number of samples for the current class based on the Dirichlet proportion
            num_samples_for_class = int(
                len(class_indices[cls]) * dirichlet_samples[client_id, cls])
            sampled_indices = class_indices[cls][:num_samples_for_class]

            # Add the selected indices to the client
            client_indices[client_id].extend(sampled_indices)

            # Update the class indices by removing the samples we assigned to this client
            class_indices[cls] = class_indices[cls][num_samples_for_class:]

    # Convert client indices into DataLoader objects
    client_loaders = [
        DataLoader(Subset(dataset, indices), batch_size=32, shuffle=True)
        for indices in client_indices.values()
    ]

    return client_loaders


# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, attacker=False):

        self.model: Net = model
        if (attacker):
            modified_loader = []
            for inputs, label in trainloader:
                modified_loader.extend(zip(inputs, (label+1) % 10))
            self.trainloader = DataLoader(
                modified_loader, batch_size=32, shuffle=True)

        else:
            self.trainloader = trainloader

        self.testloader = testloader

    def get_parameters(self, config=None):

        return [param.data.clone() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), parameters):
                param.data.copy_(torch.tensor(new_param, dtype=param.dtype))

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=0.0001, weight_decay=0.0001)

        nr_of_training_samples = 0

        for _ in range(1):

            for inputs, label in self.trainloader:

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                nr_of_training_samples += len(inputs)

        # _, _, eval_metrics = self.evaluate()

        return self.get_parameters(), nr_of_training_samples, {}

    def evaluate(self, parameters=None, config=None):

        if parameters:
            self.set_parameters(parameters)
        all_labels = []
        all_predictions = []
        all_probabilities = []

        criterion = CrossEntropyLoss()
        running_loss = 0  # Initialize running loss

        # Disable gradient calculation during evaluation for performance
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Get predictions from output
                _, predicted = torch.max(outputs, 1)

                # Store the true labels and predicted labels
                all_labels.extend(labels.numpy())
                all_predictions.extend(predicted.numpy())

                # Convert logits to probabilities (for ROC AUC)
                probabilities = F.softmax(outputs, dim=1)
                all_probabilities.extend(probabilities.numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions,
                      average='macro', zero_division=1)
        kappa = cohen_kappa_score(all_labels, all_predictions)

        # Calculate average loss over all batches
        avg_loss = running_loss / len(self.testloader)

        # Binarize the labels for ROC AUC
        all_labels_binarized = label_binarize(all_labels, classes=range(10))

        # Compute ROC AUC score for multi-class (One-vs-Rest)
        roc_auc = roc_auc_score(all_labels_binarized,
                                all_probabilities, multi_class="ovo")

        # Prepare evaluation metrics
        eval_metrics = {
            "accuracy": round(float(accuracy), 8),
            "f1_score": round(float(f1), 8),
            "kappa": round(float(kappa), 8),
            "loss": round(float(avg_loss), 8),
            "roc_auc": round(float(roc_auc), 8)
        }

        # Log the evaluation results
        with open(file_name, "a") as file:
            file.write(
                f"Evaluation: Loss: {avg_loss}, Accuracy: {accuracy}, F1 Score: {f1}, Kappa: {kappa}, ROC AUC: {roc_auc}\n"
            )

        return avg_loss, len(self.testloader), eval_metrics


def metrics_aggregation(metrics=None):
    """Metrics is tuple[int, dict[str, float]], ie. a tuple of nr of samples used
    and metric scores for each client."""

    avg_metrics = {metric: 0 for metric in metrics[0][1]}

    for _, client_metrics in metrics:
        for metric in client_metrics:
            avg_metrics[metric] += client_metrics[metric]

    num_clients = len(metrics)
    for metric in avg_metrics:
        avg_metrics[metric] = round(
            float(avg_metrics[metric] / num_clients), 8)

    return avg_metrics


def evaluate_metrics_aggregation(metrics):
    global round_int
    avg_metrics = metrics_aggregation(metrics)

    round_int += 1
    with open(file_name, "a") as file:
        file.write(f"Round: {round_int}\n\n\n")
    return avg_metrics


def generate_client_fn(trainloaders, valloaders, number_of_attckers=0):
    attackers = [i for i in range(number_of_attckers)]
    # Define the client function

    def client_fn(context: Context) -> fl.client.Client:
        client_id = int(context.node_config["partition-id"])
        return FlowerClient(Net(), trainloaders[client_id], valloaders, attacker=(client_id in attackers)).to_client()
    return client_fn


# Start the simulation
if __name__ == "__main__":
    partitioner = [dirichlet_partitioner, iid_partition_data]
    for s in range(2):
        for attackers in range(3):
            for i in range(2):
                file_name = f"log/rapport_{'FedAvg'if s == 0 else 'FedProx'}_{'Non-' if i == 0 else'' }Iid_attackers_{attackers}.txt"
                clients_data = partitioner[i](dataset, nr_of_clients)

                client_fn = generate_client_fn(clients_data, DataLoader(
                    testset, batch_size=32, shuffle=False), number_of_attckers=attackers)

                strategy = [fl.server.strategy.FedAvg(
                    min_fit_clients=1,
                    min_evaluate_clients=nr_of_clients,
                    # Minimum number of clients for training
                    min_available_clients=nr_of_clients,
                    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation), fl.server.strategy.FedProx(min_fit_clients=1,
                                                                                                              # Minimum number of clients for evaluation
                                                                                                              min_evaluate_clients=nr_of_clients,
                                                                                                              min_available_clients=nr_of_clients,      # Minimum number of clients available
                                                                                                              # Proximal term coefficient (adjust based on needs)
                                                                                                              proximal_mu=0.1,
                                                                                                              evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation
                                                                                                              )]

                fl.simulation.start_simulation(
                    client_fn=client_fn,
                    num_clients=nr_of_clients,
                    config=fl.server.ServerConfig(num_rounds=nr_of_rounds),
                    client_resources={"num_cpus": 5.0, "num_gpus": 0.0},
                    strategy=strategy[s],
                )
