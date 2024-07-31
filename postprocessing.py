import os
import cv2 
from sklearn.cluster import DBSCAN
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def detect_and_categorize_circles(img_path, epsilon, min_samples, prompt_number):
    print(f"Running detect_and_categorize_circles with epsilon: {epsilon}, min_samples: {min_samples}, and {prompt_number} prompts")
    print(img_path)
    # Read the image
    image = cv2.imread(img_path)

    image = np.array(image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Detect circles in the image
    dp = 1
    min_dist = 50
    param1 = 50
    param2 = 0.7
    min_radius = 0
    max_radius = 0
    algorithm ='brute'
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT_ALT, 
        dp=dp, 
        minDist=min_dist, 
        param1=param1, 
        param2=param2, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )

    if circles is None:
        return [], []

    radii = circles[0, :, 2].reshape(-1, 1)
    
    # Apply DBSCAN to cluster radii
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm=algorithm)
    labels = dbscan.fit_predict(radii)
    
    categorized_circles = defaultdict(list)
    for label, circle in zip(labels, circles[0, :]):
        if label != -1:  # Ignore noise points
            categorized_circles[label].append(circle)
    
    if not categorized_circles:
        return [], []
    
    most_populous_category = max(categorized_circles, key=lambda k: len(categorized_circles[k]))
    most_common_circles = categorized_circles[most_populous_category]
    
    bboxes = []
    bboxes_xy = []
    selected_circles = []
    num_circles = min(prompt_number, len(most_common_circles))
    
    def do_bounding_boxes_overlap(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def do_bounding_boxes_overlap_xy(bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

    for i in range(num_circles):
        center_x, center_y, radius = most_common_circles[i]
        x1 = center_x - radius
        y1 = center_y - radius
        x2 = center_x + radius
        y2 = center_y + radius
        width = 2 * radius
        height = 2 * radius
        bbox = [x1, y1, width, height]  # Bounding box in [x, y, width, height] format
        bbox_xy = [x1, y1, x2, y2]      # Bounding box in [x1, y1, x2, y2] format
        
        if not any(do_bounding_boxes_overlap(bbox, existing_bbox) for existing_bbox in bboxes):
            bboxes.append(bbox)
            selected_circles.append(most_common_circles[i])

        if not any(do_bounding_boxes_overlap_xy(bbox_xy, existing_bbox_xy) for existing_bbox_xy in bboxes_xy):
            bboxes_xy.append(bbox_xy)
        
    print("bboxes_xy: \n",[bboxes_xy])
    return [bboxes_xy], [bboxes]



def visualize_selected_features_and_masks(image, selected_features, selected_masks, max_display=10):
    num_selected = len(selected_features)
    num_to_display = min(num_selected, max_display)
    
    fig, axes = plt.subplots(num_to_display, 2, figsize=(12, 4 * num_to_display))
    
    for i in range(num_to_display):
        # Visualize feature map
        feature = selected_features[i].cpu().detach()
        
        # Normalize the feature map
        feature_normalized = (feature - feature.min()) / (feature.max() - feature.min())
        
        # If the feature map is 3D (C x H x W), take the mean across channels
        if feature.dim() == 3:
            feature_2d = feature_normalized.mean(0)
        else:
            feature_2d = feature_normalized.squeeze()
        
        axes[i, 0].imshow(feature_2d, cmap='viridis')
        axes[i, 0].set_title(f'Selected Feature {i+1}')
        axes[i, 0].axis('off')
        
        # Visualize mask
        mask = selected_masks[i].cpu().numpy()
        axes[i, 1].imshow(image, alpha=0.6)
        axes[i, 1].imshow(mask, alpha=0.4, cmap='jet')
        axes[i, 1].set_title(f'Selected Mask {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()



def draw_and_save_rectangles(image_path, bounding_boxes, prompt_boxes, output_dir, resize_to=518):
    """
    Draw rectangles from bounding_boxes and prompt_boxes on the image and save it.
    
    Args:
        image_path (str): Path to the image file.
        bounding_boxes (list): List of bounding boxes [x1, y1, x2, y2].
        prompt_boxes (list): List of prompt boxes [x1, y1, x2, y2].
        output_dir (str): Directory to save the output image.
        resize_to (tuple): Tuple (width, height) to resize the image. If None, no resizing is done.
    """
    image = cv2.imread(image_path)
    
    original_height, original_width = image.shape[:2]
    if resize_to:
        new_width, new_height = resize_to, resize_to
        image = cv2.resize(image, (new_width, new_height))
    else:
        new_width, new_height = original_width, original_height

    # Create a copy of the image to draw on
    output_image = image.copy()
    
    # Calculate the resize ratios
    width_ratio = new_width / original_width
    height_ratio = new_height / original_height
    
    def rescale_box(box, width_ratio, height_ratio):
        x1, y1, x2, y2 = box
        x1 = int(x1 * width_ratio)
        y1 = int(y1 * height_ratio)
        x2 = int(x2 * width_ratio)
        y2 = int(y2 * height_ratio)
        return [x1, y1, x2, y2]
    
    # Draw rectangles from bounding_boxes in green
    for box in bounding_boxes:
        if len(box[0]) != 4:
            print(box[0])
            raise ValueError("Each bounding box must be a list of 4 elements [x1, y1, x2, y2].")
        
        
        x1, y1, x2, y2 = box[0]
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle in green
    
    # Draw rectangles from prompt_boxes in red
    for box in prompt_boxes:
        if len(box) != 4:
            raise ValueError("Each prompt box must be a list of 4 elements [x1, y1, x2, y2].")
        
        rescaled_box = rescale_box(box, width_ratio, height_ratio)
        x1, y1, x2, y2 = rescaled_box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw rectangle in red
    
    # Save the modified image with rectangles drawn on it
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(output_path, output_image)
    print(f"Image with rectangles saved to {output_path}")





def extract_bbox_wh_list(masks):
    """
    Compute bounding boxes from a list of masks.
    Args:
        masks: List of binary segmentation masks (numpy arrays).
    Returns:
        List of bounding box arrays [num_instances, (x1, y1, x2, y2)].
    """
    bounding_boxes_list = []
    for mask in masks:
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]

        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            horizontal_indices = np.where(m.sum(axis=0) > 0)[0]
            vertical_indices = np.where(m.sum(axis=1) > 0)[0]
            if horizontal_indices.shape[0]:
                x1, x2 = horizontal_indices[[0, -1]]
                y1, y2 = vertical_indices[[0, -1]]
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Set bbox to zeros.
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([x1, y1, x2, y2])

        bounding_boxes_list.append(boxes.tolist())

    return bounding_boxes_list

def get_image_ids(directory_path):
    image_ids = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                # Assuming image IDs are based on filenames (without extensions)
                image_id = os.path.splitext(filename)[0]
                image_ids.append(image_id)
    return image_ids