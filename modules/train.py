
import math
import time
import numpy as np
import pickle
import os

def save_model(model, filename='model_checkpoint.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def compute_loss_and_gradient(predictions, labels):
    batch_size = len(predictions)
    loss = 0.0
    grad = []

    for pred, label in zip(predictions, labels):
        sample_loss = 0.0
        sample_grad = []
        for p, y in zip(pred, label):
            # Add small epsilon for numerical stability
            epsilon = 1e-9
            p = max(min(p, 1 - epsilon), epsilon)
            sample_loss += -y * math.log(p)
            sample_grad.append(p - y)
        loss += sample_loss
        grad.append(sample_grad)

    loss /= batch_size
    return loss, grad



import sys

import time


def train(model, train_images, train_labels, epochs=10, batch_size=64, learning_rate=0.01, checkpoint_dir="checkpoints"):
    num_samples = len(train_images)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        correct = 0
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for i in range(0, num_samples, batch_size):
            batch_images = train_images[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]

            # Forward
            output = batch_images
           
            output = model.forward(batch_images,curr_iter=i)
            # Loss + grad
            loss, grad = compute_loss_and_gradient(output, batch_labels)
            total_loss += loss

                     # Accuracy per batch
            batch_correct = sum(
                np.argmax(out) == np.argmax(label)
                for out, label in zip(output, batch_labels)
            )
            correct += batch_correct


               
            grad = model.backward(grad, learning_rate,curr_iter=i) 
            
            batch_num = i // batch_size + 1
            total_batches = (num_samples + batch_size - 1) // batch_size
            sys.stdout.write(
                    f"\r[{batch_num}/{total_batches}] "
                    f"Loss: {loss:.4f} | "
                    f"Batch Acc: {100 * batch_correct / len(batch_images):.2f}%"
            )
            sys.stdout.flush()

        duration = time.time() - start_time
        accuracy = correct / num_samples
        ips = num_samples / duration

        print(f"\nEpoch Summary - Loss: {total_loss:.4f} | Acc: {accuracy * 100:.2f}% | IPS: {ips:.2f}")

        # Save checkpoint
        save_model(model, os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pkl"))
        print(f"Checkpoint saved to {checkpoint_dir}/epoch_{epoch + 1}.pkl")
