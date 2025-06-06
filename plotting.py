import numpy as np
import matplotlib.pyplot as plt


def show_images(images, preds, out_loc, num_out=None):
    """
    Constructs an image of multiple time-series reconstruction samples compared against its relevant ground truth
    Saves locally in the given out location
    :param images: ground truth images
    :param preds: predictions from a given model
    :out_loc: where to save the generated image
    :param num_out: how many images to stack. If None, stack all
    """
    # Make sure objects are in numpy format
    if not isinstance(images, np.ndarray):
        images = images.cpu().numpy()
        preds = preds.cpu().numpy()
    
    # (32, 32, 3) -> (3, 32, 32)
    images = np.swapaxes(images, 2, 3)
    images = np.swapaxes(images, 3, 4)
    
    preds = np.swapaxes(preds, 2, 3)
    preds = np.swapaxes(preds, 3, 4)

    # Splice to the given num_out
    if num_out is not None:
        images = images[:num_out]
        preds = preds[:num_out]

    # Iterate through each sample, stacking into one image
    out_image = None
    for idx, (gt, pred) in enumerate(zip(images, preds)):
        # Pad between individual timesteps
        gt = np.pad(gt, pad_width=(
            (0, 0), (5, 5), (0, 1), (0, 0)
        ), constant_values=1)

        gt = np.hstack([i for i in gt])

        # Pad between individual timesteps
        pred = np.pad(pred, pad_width=(
            (0, 0), (0, 10), (0, 1), (0, 0)
        ), constant_values=1)

        # Stack timesteps into one image
        pred = np.hstack([i for i in pred])

        # Stack gt/pred into one image
        final = np.vstack((gt, pred))

        # Stack into out_image
        if out_image is None:
            out_image = final
        else:
            out_image = np.vstack((out_image, final))

    # Save to out location
    if images.shape[-1] == 1:
        plt.imsave(out_loc, out_image[:, :, 0], cmap='gray')
    else:
        plt.imsave(out_loc, out_image, cmap='gray')
