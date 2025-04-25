import time
import sys 
import numpy as np

def evaluate(model, test_images, test_labels):
    start_time = time.time()
    correct = 0
    total = len(test_images)

    for i in range(total):
        output = test_images[i:i+1]  # [1 x ...]
       
        output = model.forward(output, curr_iter=i)

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
