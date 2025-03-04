import os
from matplotlib.pyplot as plt

def save_reconstruction(original_image, target_image, reconstructed_image, save_dir="./figures"):
    os.makedirs(save_path, exist_ok = True)
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches((10, 5))
    ax[0].imshow(image[0][0].detach().numpy())
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    ax[1].imshow(target[0][0].detach().numpy())
    ax[1].set_title('Target Image')
    ax[1].axis('off')
    ax[2].imshow(output[0][0].detach().numpy())
    ax[2].set_title('Output Image')
    ax[2].axis('off')
    plt.savefig(os.path.join(save_path, 'reconstruction_results.png'))
    plt.show()