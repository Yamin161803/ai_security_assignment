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
