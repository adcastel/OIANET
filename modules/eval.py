import time
import sys 
"""
def evaluate(model, test_images, test_labels):
    start_time = time.time()
    correct = 0
    total = len(test_images)

    for i in range(total):
        output = test_images[i:i+1]
        start_time_b = time.time()
        for layer in model:
            output = layer.forward(output)
        durationb = time.time() - start_time_b
        print("Time for eval pass:", durationb)
        predicted = output[0].index(max(output[0]))
        actual = test_labels[i].index(1)
        if predicted == actual:
            correct += 1

        # Mini progress bar
        if i % 100 == 0 or i == total - 1:
            sys.stdout.write(f"\rEvaluating: {i+1}/{total}")
            sys.stdout.flush()

    accuracy = correct / total
    duration = time.time() - start_time
    ips = total / duration

    print(f"\nEvaluation Results - Accuracy: {accuracy * 100:.2f}% | IPS: {ips:.2f}")
"""
import time
import numpy as np
import sys

def evaluate(model, test_images, test_labels):
    start_time = time.time()
    correct = 0
    total = len(test_images)

    for i in range(total):
        output = test_images[i:i+1]  # [1 x ...]
        start_time_b = time.time()
        #for layer in model:
        #    layer_start_time = time.time()  # Start timer for the layer
        #    output = layer.forward(output)
        #    layer_time = time.time() - layer_start_time
        #    images_per_second = 1.0 / layer_time
        #    if i == 0:
        #        print(f"Layer: {layer.__class__.__name__}, Time: {layer_time:.4f}s, Performance: {images_per_second:.2f} images/sec")
        output = model.forward(output, curr_iter=i)
        durationb = time.time() - start_time_b
        print("Time for eval pass:", durationb)
        print("\n")
        # Use NumPy's argmax
        predicted = np.argmax(output[0])
        actual = np.argmax(test_labels[i])
        if predicted == actual:
            correct += 1

        # Mini progress bar
        if i % 100 == 0 or i == total - 1:
            sys.stdout.write(f"\rEvaluating: {i+1}/{total}")
            sys.stdout.flush()

    accuracy = correct / total
    duration = time.time() - start_time
    ips = total / duration

    print(f"\nEvaluation Results - Accuracy: {accuracy * 100:.2f}% | IPS: {ips:.2f}")
