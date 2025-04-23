from data.cifar100 import load_cifar100, normalize_images, one_hot_encode
from models.alexnet_cifar_100 import *
from models.resnet18_cifar_100 import ResNet18_CIFAR100
from models.tinycnn_cifar_100 import *
from models.oianet_cifar100 import OIANET_CIFAR100
from modules.train import train
from modules.eval import evaluate
model_name = 'OIANet'  # Change to 'AlexNet', 'TinyCNN', 'OIANet or 'ResNet18' as needed
performance=False
# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = load_cifar100(data_dir='/Users/adcastel/RESEARCH/OIANET/data/cifar-100-python')
train_images = normalize_images(train_images)
test_images = normalize_images(test_images)
train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

# Build and train model
if model_name == 'AlexNet':
    model = AlexNet_CIFAR100(use_im2col=False)
elif model_name == 'TinyCNN':
    model = TinyCNN(use_im2col=False)
elif model_name == 'OIANet':
    model = OIANET_CIFAR100(use_im2col=True)
else:
    model = ResNet18_CIFAR100(use_im2col=False)
bs = [8]
for b in bs:
    train(model, train_images, train_labels, epochs=10, batch_size=b, learning_rate=0.01, model_name=model_name, performance=performance)

# Evaluate model
if performance == False:
    evaluate(model, test_images, test_labels)
