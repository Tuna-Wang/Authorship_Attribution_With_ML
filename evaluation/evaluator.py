
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Evaluator:
    def __init__(self, logger):
        self.logger = logger
        
    def evaluate(self, model, test_results, prediction, ground_truth):
        """Evaluate the model's performance on the test set.

        Args:
            model: The trained model to evaluate.
            test_results: DataFrame containing the test results.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        # Compute evaluation metrics
        if test_results is not None and not test_results.empty: 
            accuracy = accuracy_score(test_results['true_label'], test_results['predicted_label'])
            self.logger.info(f'Accuracy: {accuracy:.4f}')
                
            # Convert classes to strings for target_names
            target_names = [str(cls) for cls in model.classes_]
            classification_rep = classification_report(test_results["true_label"], test_results["predicted_label"], target_names=target_names)
            self.logger.info(f'Classification Report:\n{classification_rep}')

            # Compute confusion matrix
            conf_matrix = confusion_matrix(test_results['true_label'], test_results['predicted_label'])
            self.logger.info(f'Confusion Matrix:\n{conf_matrix}')


        else:
            accuracy = accuracy_score(ground_truth, prediction)
            self.logger.info(f'Accuracy: {accuracy:.4f}')
            classification_rep = classification_report(ground_truth, prediction)
            self.logger.info(f'Classification Report:\n{classification_rep}')
            conf_matrix = confusion_matrix(ground_truth, prediction)
            self.logger.info(f'Confusion Matrix:\n{conf_matrix}')   

        return {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix
        }