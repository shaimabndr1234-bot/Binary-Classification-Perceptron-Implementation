import numpy as np

def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            features = list(map(float, values[:-1]))
            label = int(values[-1].split('-')[-1])
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

def filter_classes(data, labels, class1, class2):
    indices = np.where((labels == class1) | (labels == class2))
    filtered_data = data[indices]
    filtered_labels = labels[indices]
    filtered_labels = np.where(filtered_labels == class1, 1, -1)  # Binary labels
    return filtered_data, filtered_labels

def perceptron_train(data, labels, iterations):
    weights = np.zeros(data.shape[1])
    bias = 0
    for _ in range(iterations):
        for x, y in zip(data, labels):
            prediction = np.sign(np.dot(weights, x) + bias)
            if prediction != y:
                weights += y * x
                bias += y
    return weights, bias

def perceptron_predict(data, weights, bias):
    raw = np.dot(data, weights) + bias
    preds = np.sign(raw)
    preds[preds == 0] = 1
    return preds


def accuracy(predictions, labels):
    return np.mean(predictions == labels) * 100

def main():
    train_data, train_labels = load_data('train.data')
    test_data, test_labels = load_data('test.data')

    class_pairs = [(1, 2), (2, 3), (1, 3)]
    for class1, class2 in class_pairs:
        print(f"\nTraining classifier for class {class1} vs class {class2}")
        
        train_x, train_y = filter_classes(train_data, train_labels, class1, class2)
        test_x, test_y = filter_classes(test_data, test_labels, class1, class2)
        
        weights, bias = perceptron_train(train_x, train_y, iterations=20)
        
        train_predictions = perceptron_predict(train_x, weights, bias)
        test_predictions = perceptron_predict(test_x, weights, bias)
        
        train_acc = accuracy(train_predictions, train_y)
        test_acc = accuracy(test_predictions, test_y)
        
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        most_discriminative_feature = np.argmax(np.abs(weights))
        print(f"Most discriminative feature: Feature {most_discriminative_feature + 1}")

if __name__ == "__main__":
    main()