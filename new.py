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
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# Global variables
server_address = "localhost:3000"
nr_of_rounds = 3
nr_of_clients = 5
client_id_file = "client_ids.txt"


# Get datasets
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.CIFAR10(
    root='./datasets', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(
    root='./datasets', train=False, download=True, transform=transform)

# Split dataset for clients for IID and Non-IID scenarios


def partition_data(dataset, num_clients):
    partitions = random_split(
        dataset, [len(dataset) // num_clients] * num_clients)
    return [DataLoader(part, batch_size=32, shuffle=True) for part in partitions]



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
    def __init__(self, model, trainloader, testloader, context, client_id):
        self.model: Net = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.context = context
        self.id = client_id

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

        _, _, eval_metrics = self.evaluate()

        return self.get_parameters(), nr_of_training_samples, eval_metrics

    def evaluate(self, parameters=None, config=None):

        if parameters:
            self.set_parameters(parameters)

        all_labels = []
        all_predictions = []

        criterion = CrossEntropyLoss()
        running_loss = 0        # Running loss

        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.numpy())
                all_predictions.extend(predicted.numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(
            all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')

        nr_of_test_samples = len(all_predictions)
        # eval_metrics = {"accuracy": accuracy, "precision": precision, "recall": recall}

        eval_metrics = {"accuracy": round(float(accuracy), 8), "precision": round(
            float(precision), 8), "recall": round(float(recall), 8)}

        with open("log.txt", "a") as file:
            file.write(
                f"Client {self.id} Evaluate: {running_loss / nr_of_test_samples, nr_of_test_samples, eval_metrics}\n")

        return running_loss / nr_of_test_samples, nr_of_test_samples, eval_metrics


def fit_metrics_aggregation(metrics):
    avg_metrics = metrics_aggregation(metrics)
    with open("log.txt", "a") as file:
        file.write(f"Aggregate fit metrics.\n")
    return avg_metrics


def evaluate_metrics_aggregation(metrics):
    avg_metrics = metrics_aggregation(metrics)
    with open("log.txt", "a") as file:
        file.write(f"Aggregate evaluate metrics.\n")
    return avg_metrics


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


def generate_client_fn(trainloaders, valloaders):

    # Define the client function
    def client_fn(context: Context) -> fl.client.Client:

        client_id = context.node_config["partition-id"]

        with open("log.txt", "a") as file:
            file.write(f"Created client {client_id}\n")

        return FlowerClient(Net(), trainloaders[int(client_id)], valloaders, context, int(client_id)).to_client()
    return client_fn


# Start the simulation
if __name__ == "__main__":
    clients_data = partition_data(dataset, nr_of_clients)
    client_fn = generate_client_fn(clients_data, DataLoader(
        testset, batch_size=32, shuffle=False))

    strategy = (fl.server.strategy.FedAvg(
        min_fit_clients=nr_of_clients,
        min_evaluate_clients=nr_of_clients,
        min_available_clients=nr_of_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        fit_metrics_aggregation_fn=fit_metrics_aggregation
    ))

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=nr_of_clients,
        config=fl.server.ServerConfig(num_rounds=nr_of_rounds),
        client_resources={"num_cpus": 4.5, "num_gpus": 0.0},
        strategy=strategy,
    )