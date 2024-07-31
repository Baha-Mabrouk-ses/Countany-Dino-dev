import numpy as np
import cv2
from skimage import segmentation, measure
from PIL import Image
#import scipy.ndimage as ndimage
import torch
from skimage import morphology


@torch.jit.script
def mask_iou(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
) -> torch.Tensor:

    """
    Inputs:
    mask1: BxNxHxW torch.float32. Consists of [0, 1]
    mask2: BxMxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: BxNxM torch.float32. Consists of [0 - 1]
    """

    B, N, H, W = mask1.shape
    B, M, H, W = mask2.shape

    mask1 = mask1.view(B, N, H * W)
    mask2 = mask2.view(B, M, H * W)

    intersection = torch.matmul(mask1, mask2.swapaxes(1, 2))

    area1 = mask1.sum(dim=2).unsqueeze(1)
    area2 = mask2.sum(dim=2).unsqueeze(1)

    union = (area1.swapaxes(1, 2) + area2) - intersection

    ret = torch.where(
        union == 0,
        torch.tensor(0.0, device=mask1.device),
        intersection / union,
    )

    return ret


@torch.jit.script
def filter_proposal(proposal_masks: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Inputs:
    proposal_masks: 1xNxHxW torch.float32. Consists of [0, 1]
    iou_threshold: float. IoU threshold for filtering proposals.
    Outputs:
    filtered_proposal_masks: Bx(N-f)xHxW torch.float32. Filtered proposal masks.
    """
    B, N, H, W = proposal_masks.shape
    
    # Compute pairwise IoU between all proposal masks
    iou_matrix = mask_iou(proposal_masks, proposal_masks)

    # Initialize a boolean mask to filter proposal masks
    mask = torch.ones(N, dtype=torch.bool, device=proposal_masks.device)

    # Check for each proposal mask if there is another mask with IoU exceeding the threshold
    for i in range(N):
        for j in range(i + 1, N):
            if ((iou_matrix.squeeze() if iou_matrix.dim() == 3 else iou_matrix)[i, j] > iou_threshold):
                mask[j] = False
                break

    # Filter proposal masks
    filtered_proposal_masks = proposal_masks[:, mask, :, :]

    return filtered_proposal_masks

def pad_transform(img, bboxes):
    """
    Pads the image to a square with the longest edge as the side length and transforms the bounding boxes accordingly.
    Args:
        img (PIL.Image): Input image.
        bboxes (list of lists): List of bounding boxes, each represented as a list [x_min, y_min, x_max, y_max].
    Returns:
        PIL.Image: Padded image.
        list of lists: List of transformed bounding boxes.
    """
    width, height = img.size

    # Calculate padding
    max_dim = max(width, height)
    pad_width = max_dim - width
    pad_height = max_dim - height

    # Create a new blank image with padding
    padded_img = Image.new("RGB", (max_dim, max_dim), color=(0, 0, 0))
    padded_img.paste(img, (pad_width // 2, pad_height // 2))

    # Transform bounding boxes
    transformed_bboxes = []
    for box in bboxes:
        x_min, y_min, x_max, y_max = box
        new_x_min = x_min + pad_width // 2
        new_y_min = y_min + pad_height // 2
        new_x_max = x_max + pad_width // 2
        new_y_max = y_max + pad_height // 2
        transformed_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
    return padded_img, transformed_bboxes

def mask_to_bbox(mask):
    if mask.dim() != 2:
        raise ValueError("mask must be a 2D binary tensor")

    # Find the indices where mask is non-zero
    horizontal_indices = torch.where(mask.any(dim=0))[0]
    vertical_indices = torch.where(mask.any(dim=1))[0]

    if horizontal_indices.numel() > 0 and vertical_indices.numel() > 0:
        x1, x2 = horizontal_indices[[0, -1]]
        y1, y2 = vertical_indices[[0, -1]]
        x2 += 1  # Make sure to include the last pixel
        y2 += 1  # Make sure to include the last pixel
    else:
        return [0, 0, 0, 0]
    return [x1.item(), y1.item(), x2.item(), y2.item()]


def preprocess_roi_mask(mask):#, min_size=50):
    """
    Preprocess the mask to remove small objects and return the largest connected component.
    Args:
        mask (torch.Tensor): Binary segmentation mask (torch tensor).
        min_size (int): Minimum size of objects to keep.
    Returns:
        torch.Tensor: Processed mask with the largest connected component.
    """
    if mask.dim() != 2:
        raise ValueError("mask must be a 2D binary tensor")

    # Convert tensor mask to ndarray
    mask_np = mask.numpy().astype(np.int8)

    # Remove objects touching the border
    mask_np = segmentation.clear_border(mask_np)
    
    # Remove small objects
    #mask_np = morphology.remove_small_objects(mask_np.astype(bool), min_size=min_size).astype(np.int8)

    # Label connected components
    labels = measure.label(mask_np, connectivity=2)
    
    # Find the largest connected component (excluding background)
    if labels.max() == 0:  # No connected components
        return torch.zeros_like(mask, dtype=torch.float32)
    
    largest_cc = labels == np.argmax(np.bincount(labels.flat, weights=mask_np.flat))
    
    processed_mask = torch.tensor(largest_cc, dtype=torch.float32)

    # Add mask dilation if needed
    # processed_mask = torch.tensor(morphology.dilation(largest_cc), dtype=torch.float32)

    return processed_mask


def build_slic_point_grid(image, crop_box=None, n_segments=100, compactness=10.0, max_num_iter=10, sigma=0, spacing=None, enforce_connectivity=True, slic_zero=True):
    """Generates a 2D grid of points based on SLIC superpixels."""
    if crop_box is not None:
        crop_box = np.array(crop_box)
        image = image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    labels = segmentation.slic(image, n_segments=n_segments, compactness=compactness, max_num_iter=max_num_iter, sigma=sigma, spacing=spacing, enforce_connectivity=enforce_connectivity)
    props = measure.regionprops(labels)
    centroids = np.array([prop.centroid for prop in props])
    # Normalize centroids to [0, 1]
    centroids /= np.array([image.shape[1], image.shape[0]])[:, None]
    return centroids


def build_all_layer_slic_point_grids(image, crop_boxes, n_segments=100, compactness=10.0, max_num_iter=10, sigma=0, spacing=None, enforce_connectivity=True, slic_zero=True):
    """Generates point grids for all crop layers based on SLIC superpixels."""
    point_grids = []
    for crop_box in crop_boxes:
        point_grid = build_slic_point_grid(image, crop_box, n_segments=n_segments, compactness=compactness, max_num_iter=max_num_iter, sigma=sigma, spacing=spacing, enforce_connectivity=enforce_connectivity, slic_zero=slic_zero)
        point_grids.append(point_grid)
    return point_grids