import torch
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class EmbeddingsClassifier:
    def __init__(self, pretrained_path: str | None = None):
        self._scaler = StandardScaler()
        if not pretrained_path:
            self._classifier = LogisticRegression(max_iter=1000)
        else:
            pass

    def train_classifier(self, X, y, save_path: str):
        # Normalise
        X_norm = self._scaler.fit_transform(X)
        self._classifier.fit(X_norm, y)

        with open(save_path, 'wb') as f:
            pickle.dump(self._classifier, f)

    def evaluate(self, X, y):
        X_norm = self._scaler.fit_transform(X)
        predictions = self._classifier.predict(X_norm)
        return accuracy_score(y, predictions)


if __name__ == '__main__':
    train_set = torch.load('datasets/encoded/simclr/encoded_cifar10_train.pt', weights_only=False)
    test_set = torch.load('datasets/encoded/simclr/encoded_cifar10_test.pt', weights_only=False)

    train_features = train_set['encodings'].numpy()
    train_labels = train_set['labels'].numpy()
    test_features = test_set['encodings'].numpy()
    test_labels = test_set['labels'].numpy()

    model = EmbeddingsClassifier()
    model.train_classifier(
        train_features,
        train_labels,
        save_path='datasets/classifiers/logreg-simclr-b256-embeddings-model'
    )

    test_accuracy = model.evaluate(test_features, test_labels)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
