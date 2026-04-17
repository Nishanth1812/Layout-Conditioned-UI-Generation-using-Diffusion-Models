from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def visualize_layout_tensor(tensor):
    # tensor: (10,64,64)
    grid = np.sum(tensor, axis=0)

    plt.imshow(grid, cmap='viridis')
    plt.title("Layout Heatmap")
    plt.colorbar()
    plt.show()


def compare_images(real_path, generated_path):
    real = Image.open(real_path)
    gen = Image.open(generated_path)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(real)
    plt.title("Real")

    plt.subplot(1,2,2)
    plt.imshow(gen)
    plt.title("Generated")

    plt.show()


