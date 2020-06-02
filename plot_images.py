import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io

def plot_image_strips(shot_images, generated_images, generated_angles, ground_truth_images, ground_truth_angles, image_height, image_width, angles_to_plot, output_path):
    canvas_width = 14 # 1 shot image + 1 space + 12 images spaced every 30 degrees
    canvas_height = 2 # 1 row of generated images + 1 row of ground truth iamges
    canvas = np.empty((image_height * canvas_height, image_width * canvas_width))

    generated_images = np.array([generated_images[0].cpu()[np.where(generated_angles[0][:, 0] == angle)[0]]
                        for angle in angles_to_plot[:, 0]])

    ground_truth_images = np.array([ground_truth_images[np.where(ground_truth_angles[:, 0] == angle)[0]]
                            for angle in angles_to_plot[:, 0]])
    # order images by angle
    generated_images = generated_images[np.argsort(angles_to_plot[:, 0], 0)]
    ground_truth_images = ground_truth_images[np.argsort(angles_to_plot[:, 0], 0)]

    blank_image = np.ones(shape=(image_height, image_width))

    # plot the first row which consists of: 1 shot image, 1 blank, 12 generated images equally spaced 30 degrees in azimuth
    # plot the shot image
    canvas[0:image_height, 0:image_width] = shot_images[0].squeeze()

    # plot 1 blank
    canvas[0:image_height, image_width:2 * image_width] = blank_image

    # plot generated images
    image_index = 0
    for column in range(2, canvas_width):
        canvas[0:image_height, column * image_width:(column + 1) * image_width] = generated_images[image_index].detach().numpy().squeeze()
        image_index += 1

    # plot the ground truth strip in the 2nd row
    # Plot 2 blanks
    k = 0
    for column in range(0, 2):
        canvas[image_height:2 * image_height, column * image_width:(column + 1) * image_width] = blank_image
    # Plot ground truth images
    image_index = 0
    for column in range(2, canvas_width):
        canvas[image_height:2 * image_height, column * image_width:(column + 1) * image_width] = ground_truth_images[image_index].squeeze()
        image_index += 1

    #for i in range(len(canvas)):
    #   canvas[i] = canvas[i].cpu().detach().numpy()
    #canvas = canvas[0][0][0]

    plt.figure(figsize=(8, 10), frameon=False)
    plt.axis('off')
    plt.imshow(canvas, origin='upper', cmap='gray')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return canvas
