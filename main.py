from data.cifar100 import load_cifar100, normalize_images, one_hot_encode
from models.alexnet_cifar_100 import *
from models.resnet18_cifar_100 import ResNet18_CIFAR100
from models.tinycnn_cifar_100 import *
from models.oianet_cifar100 import OIANET_CIFAR100
from modules.train import train
from modules.eval import evaluate
from modules.performance import perf

def main(model_name, batch_size, epochs, learning_rate, conv_algo, performance):
    (train_images, train_labels), (test_images, test_labels) = load_cifar100(data_dir='./data/cifar-100-python')
    train_images = normalize_images(train_images)
    test_images = normalize_images(test_images)
    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    # Build and train model
    if model_name == 'AlexNet':
        model = AlexNet_CIFAR100(conv_algo=conv_algo)
    elif model_name == 'TinyCNN':
        model = TinyCNN(conv_algo=conv_algo)
    elif model_name == 'OIANet':
        model = OIANET_CIFAR100(conv_algo=conv_algo)
    else:
        model = ResNet18_CIFAR100(conv_algo=conv_algo)

    if performance:
        perf(model, train_images, train_labels, batch_size=batch_size)
    else:
        train(model, train_images, train_labels, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
              save_path=f'saved_models/{model_name}', resume=True)
        evaluate(model, test_images, test_labels)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Train a CNN model on CIFAR-100.')
    parser.add_argument('--model', type=str, choices=['AlexNet', 'TinyCNN', 'OIANet', 'ResNet18'], default='OIANet',
                        help='Model to train (default: OIANet)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training (default: 0.01)')
    parser.add_argument('--performance', action='store_true', help='Enable performance measurement')
    parser.add_argument('--conv_algo', type=int, default=0, choices=[0,1,2], help='Conv2d algorithm 0-direct, 1-im2col, 2-im2colfused (default: 0)')
    args = parser.parse_args()
    model_name = args.model
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    performance = args.performance
    conv_algo = args.conv_algo
    
    main(model_name, batch_size, epochs, learning_rate, conv_algo, performance)