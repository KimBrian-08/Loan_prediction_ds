class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates model performance using various metrics.
        :param true_labels: Actual labels (ground truth)
        :param predicted_labels: Labels predicted by the model
        :return: Dictionary containing evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def compare_models(self, model_results):
        """
        Compares multiple models based on their evaluation metrics.
        :param model_results: Dictionary where keys are model names and values are their evaluation metrics
        :return: Sorted list of models based on F1 Score
        """
        return sorted(model_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)