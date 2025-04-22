from data.cifar100 import load_cifar100, normalize_images, one_hot_encode
from models.alexnet_cifar_100 import build_alexnet_for_cifar100
from modules.train import train
from modules.eval import evaluate

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = load_cifar100(data_dir='/Users/adcastel/RESEARCH/OIA_PYTHON/data/cifar-100-python')
train_images = normalize_images(train_images)
test_images = normalize_images(test_images)
train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

# Build and train model
model = build_alexnet_for_cifar100()
train(model, train_images, train_labels, epochs=10, batch_size=8, learning_rate=0.01)

# Evaluate model
evaluate(model, test_images, test_labels)
