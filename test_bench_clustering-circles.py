import cv2
import argparse
import csv
import json
import numpy as np
from tqdm import tqdm
from os.path import exists
import os
import random
import datetime
import logging
from segment_anything import sam_model_registry as sam_registry
from mobile_sam import sam_model_registry as mobile_sam_registry
from countanything.countanything import SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import time
import wandb
import torch
from collections import defaultdict
from sklearn.cluster import DBSCAN


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        x0, y0, w, h = ann['bbox']
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.scatter([x0+w//2], [y0+h//2], color='green', marker='*', s=10, edgecolor='white', linewidth=1.25)


def draw_and_save_rectangles(image_path, bounding_boxes, output_dir):

    image = cv2.imread(image_path)
    
    # Create a copy of the image to draw on
    output_image = image.copy()
    
    # Draw rectangles on the image
    for box in bounding_boxes:
        if len(box) != 4:
            raise ValueError("Each bounding box must be a list of 4 elements [x, y, w, h].")
        
        x, y, w, h = map(int, box)  # Ensure coordinates are integers
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
    
    # Save the modified image with rectangles drawn on it
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(output_path, output_image)
    print(f"Image with rectangles saved to {output_path}")


def draw_and_save_circles_and_boxes(image_path, circles, boxes, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    
    # Create a copy of the image to draw on
    output_image = image.copy()
    
    # Draw circles on the image
    for circle in circles:
        center_x, center_y, radius = circle
        cv2.circle(output_image, (center_x, center_y), radius, (0, 255, 0), 2)  # Draw circle
    
    # Draw bounding boxes on the image
    for box in boxes:
        x, y, width, height = box
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        cv2.rectangle(output_image, top_left, bottom_right, (255, 0, 0), 2)  # Draw bounding box
    
    # Save the modified image with circles and boxes drawn on it
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(output_path, output_image)
    print(f"Image with circles and boxes saved to {output_path}")

def show_prompt(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        x0, y0, w, h = ann['bbox']
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.scatter([x0+w//2], [y0+h//2], color='green', marker='*', s=10, edgecolor='white', linewidth=1.25)

def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    Arguments:
    bbox1: List representing the coordinates of the first bounding box in the format [x, y, width, height].
    bbox2: List representing the coordinates of the second bounding box in the format [x, y, width, height].
    
    Returns:
    iou: Intersection over Union (IoU) value between the two bounding boxes.
    """
    # Convert bbox1 to [x1, y1, x2, y2] format
    x1_bbox1, y1_bbox1, x2_bbox1, y2_bbox1 = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    # Convert bbox2 to [x1, y1, x2, y2] format
    x1_bbox2, y1_bbox2, x2_bbox2, y2_bbox2 = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
    
    # Calculate the coordinates of the intersection rectangle
    x1_intersection = max(x1_bbox1, x1_bbox2)
    y1_intersection = max(y1_bbox1, y1_bbox2)
    x2_intersection = min(x2_bbox1, x2_bbox2)
    y2_intersection = min(y2_bbox1, y2_bbox2)
    
    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(0, y2_intersection - y1_intersection + 1)
    
    # Calculate the area of each bounding box
    area_b1 = (x2_bbox1 - x1_bbox1 + 1) * (y2_bbox1 - y1_bbox1 + 1)
    area_b2 = (x2_bbox2 - x1_bbox2 + 1) * (y2_bbox2 - y1_bbox2 + 1)
    
    # Calculate the Union area
    union_area = area_b1 + area_b2 - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou



def detect_and_categorize_circles(img_path, epsilon, min_samples, prompt_number, max_radius):
    print(f"Running detect_and_categorize_circles with epsilon: {epsilon}, min_samples: {min_samples}, and {prompt_number} prompts")

    # Read the image
    image = cv2.imread(img_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    

 
    # Detect circles in the image
    dp = 1
    min_dist = 20
    param1 = 50
    param2 = 0.8
    min_radius = 8
    max_radius = 200
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

    print(f"Detected circles: {circles}")
    # Check if any circles are detected
    if circles is None:
        return [], []

    circles = np.uint16(np.around(circles))
    
    # Extract radii from circles
    radii = circles[0, :, 2].reshape(-1, 1)
    
    # Apply DBSCAN to cluster radii
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(radii)
    
    # Categorize circles based on DBSCAN labels
    categorized_circles = defaultdict(list)
    for label, circle in zip(labels, circles[0, :]):
        if label != -1:  # Ignore noise points
            categorized_circles[label].append(circle)
    
    # Find the most populous category
    if not categorized_circles:
        return [], []
    
    most_populous_category = max(categorized_circles, key=lambda k: len(categorized_circles[k]))
    most_common_circles = categorized_circles[most_populous_category]
    
    # Find the center of the image
    img_center_x = image.shape[1] // 2
    img_center_y = image.shape[0] // 2
    
    # Calculate the distance of each circle from the image center
    def distance_from_center(circle):
        center_x, center_y, _ = circle
        return np.sqrt((center_x - img_center_x) ** 2 + (center_y - img_center_y) ** 2)
    
    # Sort circles by their distance from the image center
    most_common_circles.sort(key=distance_from_center)
    
    # Create bounding boxes for the closest circles to the center
    bboxes = []
    num_circles = min(prompt_number, len(most_common_circles))
    
    for i in range(num_circles):
        center_x, center_y, radius = most_common_circles[i]
        x = center_x - radius
        y = center_y - radius
        width = 2 * radius
        height = 2 * radius
        bbox = [x, y, width, height]  # Bounding box in [x, y, width, height] format
        bboxes.append(bbox)
    
    return bboxes, most_common_circles


# Combine both registries of mobile_sam and vanilla sam in one
combined_registry = {**mobile_sam_registry, **sam_registry}

parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp",  "--data_path", type=str, default='counting_pipe.v1i.coco/test/images', help="Path to the FSC147 dataset")
parser.add_argument("-mt",  "--model_type", type=str, default="vit_l", help="model type")
parser.add_argument("-mp",  "--model_path", type=str, default="Models/sam_vit_h_4b8939.pth", help="path to trained model")
parser.add_argument("-op",  "--output_path", type=str, default="Benchmark_Result_default_path", help="path to Output")
parser.add_argument("-io",  "--test_iou_threshold", type=float, default=0.75, help="Set testing Iou Thresh'hold")
parser.add_argument("-ip",  "--pred_iou_threshold", type=float, default=0.8, help="Set AutoMaskGen prediction Iou Thresh'hold")
parser.add_argument("-st",  "--sim_thresh", type=float, default=0.8, help="Set Similarity Thresh'hold")
parser.add_argument("-np",  "--prompt_num", type=int, default=3, help="chose number of prompts")
parser.add_argument("-p1",  "--param1", type=int, default=75, help=" Higher threshold for the Canny edge detector")
parser.add_argument("-p2",  "--param2", type=float, default=0.85, help=" confidence")
parser.add_argument("-eps",  "--epsilon", type=int, default=20, help=" epsilon")
parser.add_argument("-ms",  "--min_samples", type=int, default=3, help=" min_samples")
parser.add_argument("-v",   "--viz", type=bool, default=True, help="whether to visualize")
parser.add_argument("-d",   "--device", default='0', help='assign device')

args = parser.parse_args()


pred_iou_thresh = args.pred_iou_threshold
sim_thresh = args.sim_thresh
data_path = args.data_path
anno_file = os.path.join(data_path, 'merged_annotations.json')
im_dir = os.path.join(data_path, 'imgs')
test_iou_threshold = args.test_iou_threshold

benchmark_dir = args.output_path

os.makedirs(benchmark_dir, exist_ok=True)

prompt_type="hough_circle"
model_name=args.model_type
num_files = len(os.listdir(im_dir))
prompt_number=args.prompt_num

param1 = args.param1            #Higher threshold for the Canny edge detector
param2 = args.param2            #Confidence

epsilon=args.epsilon            #similarity per cluster
min_samples= args.min_samples   #minimum objects per cluster

debug = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
device = 'cuda'
sam = combined_registry[args.model_type](checkpoint=args.model_path)
sam.to(device=device)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="countany+Dino",
    # track hyperparameters and run metadata
    config={
    f"Iou_thresh": {pred_iou_thresh},
    f"sim_ thresh": {sim_thresh},
    f"Number of images": {num_files},
    f"Model": {model_name},
    f"prompting":{prompt_type},
    f"number of prompts":{prompt_number},
    f"confidence":{param2}
    }      
)


if not exists(anno_file) or not exists(im_dir):
    print(anno_file, im_dir)
    print("Make sure you set up the --data-path correctly.")
    print("Current setting is {}, but the image dir and annotation file do not exist.".format(args.data_path))
    print("Aborting the evaluation")
    exit(-1)


mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=25 , sim_thresh=sim_thresh, pred_iou_thresh=pred_iou_thresh )

with open(anno_file) as f:
    annotations = json.load(f)


# Generate timestamp for unique directory names and file names
timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")

# Directory path for individual batch benchmark results
indiv_benchmark_dir = os.path.join(benchmark_dir, f"hough_Confidence_{param2}_param1_{param1}__{model_name}_prompt_{prompt_type}_{prompt_number}")

# Create the individual batch benchmark directory
os.makedirs(indiv_benchmark_dir, exist_ok=True)

# Define the CSV file path for individual image metrics
csv_file_object = os.path.join(indiv_benchmark_dir, f"Object_lvl_hough_Confidence_{param2}_param1_{param1}__{model_name}_prompt_{prompt_type}_{prompt_number}.csv")

# Define the CSV file path for individual image metrics
csv_file_individual = os.path.join(indiv_benchmark_dir, f"Image_lvl_hough_Confidence_{param2}_param1_{param1}__{model_name}_prompt_{prompt_type}_{prompt_number}.csv")

# Define the CSV file path for aggregate results
csv_file_aggregate = os.path.join(benchmark_dir,f"results_clustering_closeToCenter.csv")



sum_overall_error = 0
sum_err = 0 
sum_MAE = 0 
sum_RMSE = 0 
sum_accuracy = 0
sum_precision = 0 
sum_recall = 0 
sum_F1_score = 0 
sum_exec = 0
sum_FN_percent =0
sum_FP_percent = 0 
sum_percent_err = 0
sum_sqr_err = 0
sum_pred_cnt = 0
sum_gt_cnt = 0


min_radius =  10           # Minimum circle radius in pixels
max_radius = 120          # Maximum circle radius in pixels
min_dist = 10           # Minimum distance between circle centers in pixels
dp = 1                   # Inverse ratio of accumulator resolution to image resolution
                        #confidence
image_name=""


# Open the CSV file for individual image metrics in write mode
with open(csv_file_object, mode='w', newline='') as file_object:
    writer_object = csv.writer(file_object)
    object_file_empty = os.stat(csv_file_object).st_size == 0

    if object_file_empty:
        writer_object.writerow(['model name','image name', 'pred_vs_ann_iou',' mask generator pred_iou',
                                 'mask generator stability score'])


    # Open the CSV file for individual image metrics in write mode
    with open(csv_file_individual, mode='w', newline='') as file_individual:
        writer_individual = csv.writer(file_individual)
        writer_individual.writerow(['model_name','image_name', 'generator_execution_time' ,
                                    'pred_cnt','GT_Count', 'TP', 'count Error', 'detect error' ,
                                    'Iou_avg', 'FN' , 'FP', 'precision', 'overall_error', 'recall', 'F1_Score'])

        for img_file in tqdm(os.listdir(im_dir)):
            
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(im_dir, img_file)
            image_id = None
            bboxes_gt=[]
            for img_info in annotations["images"]:
                if img_info["file_name"] == img_file:
                    image_id = img_info["id"]
                    image_name = img_info["file_name"]
                    break
            

            if image_id is not None:
                for annotation in annotations["annotations"]:
                    if annotation["image_id"] == image_id:
                        bboxes_gt.append(annotation["bbox"])
                                    
            
            try:
                bounding_boxes, circles_to_draw = detect_and_categorize_circles(
                    img_path,epsilon, min_samples, prompt_number
                    )
                # Add any further processing you need here
            except Exception as e:
                print(f"An error occurred while processing {img_path}: {e}")
                continue  # Skip to the next iteration

            if len(bounding_boxes) == 0: 
                print("bounding_boxes is empty")
                continue

            input_boxes = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in bounding_boxes] 

            viz_box= [[bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in bounding_boxes] 


            input_boxes = np.array(input_boxes, dtype=np.float64)

            print(len(input_boxes))

            image = cv2.imread(img_path)

            start_time_generator = time.time()

            print("input_boxes before generate:", input_boxes)

            masks = mask_generator.generate(image, input_boxes)

            end_time_generator = time.time()

            generator_execution_time = end_time_generator - start_time_generator


            start_time_viz = time.time()

            if viz:
                hough_prompt_viz=str(indiv_benchmark_dir+"/prompt_viz")
                hough_bbox_viz=str(indiv_benchmark_dir+"/bbox_viz")
    
                if not exists(hough_bbox_viz):
                    os.mkdir(hough_bbox_viz)
                
                draw_and_save_circles_and_boxes(img_path, circles_to_draw, bounding_boxes, hough_prompt_viz)
    
    
                if not exists(indiv_benchmark_dir):
                    os.mkdir(indiv_benchmark_dir)
    
                
                plt.figure(figsize=(10,10))
                plt.imshow(image)
                show_anns(masks)
                plt.axis('off')
                plt.savefig(os.path.join(indiv_benchmark_dir, img_file))
                plt.close()

            end_time_viz = time.time()

            gt_cnt = len(bboxes_gt)
            pred_cnt = len(masks)


            FN = 0  # False Negatives 
            TP = 0  # True positives
            FP= 0   # False Positives
            Iou_sum = 0 # summing iou to be averaged
            Iou_avg = 0 # avg iou for single image 
            
            for bbox_gt in bboxes_gt:
                max_iou = 0  # Initialize maximum IoU
                max_iou_bbox = None  # Initialize bbox_pred with maximum IoU
                
                for bbox_pred in masks:
                    
                    mask_iou = calculate_iou(bbox_gt, bbox_pred['bbox'])

                    # mask_exist_condition= False # know if the predicted mask has a corresponding annontation, if not it counts as a false negative
                    

                    if mask_iou > max_iou:
                        mask_exist_condition= True
                        max_iou = mask_iou
                        max_iou_bbox = bbox_pred
                        pred_iou = bbox_pred['predicted_iou']
                        stability_score = bbox_pred['stability_score']

                    #if not mask_exist_condition :
                    #    FP+=1  
    
                
                if max_iou >= test_iou_threshold:
                    
                    Iou_sum += max_iou
                    TP += 1
                    pred_vs_ann_iou=max_iou

                    writer_object.writerow([model_name , image_name , pred_vs_ann_iou, pred_iou, stability_score])

            FP = pred_cnt - TP 
            FN = gt_cnt-TP if gt_cnt>TP else 0
           
            count_err = abs(gt_cnt - pred_cnt)
            detect_err = abs(pred_cnt- TP) 
            overall_error = count_err + FP + FN
            overall_error_percent = 100*overall_error/gt_cnt if gt_cnt>0 else 0
            percent_detect_err = 100*detect_err/TP if TP>0 else 0
            sqr_err = abs(gt_cnt - pred_cnt)**2 if gt_cnt !=0 else 0
            percent_err = count_err/gt_cnt if gt_cnt !=0 else 0
            Iou_avg = Iou_sum/TP if TP !=0 else 0
            accuracy = TP/(FP +TP+ FN) if (FP +TP) != 0 else 0
            precision = TP / pred_cnt if pred_cnt > 0 else 0
            recall = TP / gt_cnt if gt_cnt > 0 else 0
            F1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

                
            
            sum_percent_err += percent_err
            sum_err += count_err
            sum_sqr_err += sqr_err
            sum_precision += precision
            sum_accuracy += accuracy
            sum_overall_error += overall_error_percent
            sum_recall += recall
            sum_F1_score += F1_score
            sum_exec += generator_execution_time
            sum_pred_cnt += pred_cnt
            sum_gt_cnt += gt_cnt
            sum_FN_percent += 100*FN/pred_cnt
            sum_FP_percent += 100*FP/pred_cnt
          
            writer_individual.writerow([model_name]+ [image_name, generator_execution_time,pred_cnt, gt_cnt, TP, count_err ,detect_err, Iou_avg, FN , FP , precision, overall_error, recall ,F1_score])
    


FP_rate =sum_FP_percent/num_files
FN_rate =sum_FN_percent/num_files
avg_overall_error = sum_overall_error/num_files
overall_accuracy = sum_accuracy/num_files
MAPE = sum_percent_err/num_files
MAE = sum_err/num_files
MSE = sum_sqr_err/num_files
RMSE = MSE**0.5
avg_precision = sum_precision/num_files
avg_recall = sum_recall/num_files
avg_F1_score = sum_F1_score/num_files
avg_exec = sum_exec/num_files

metrics_dict = {
    "FP_rate": FP_rate,
    "FN_rate": FN_rate,

    "avg_overall_error": avg_overall_error,
    "overall_accuracy": overall_accuracy,
    "MAPE": MAPE,
    "MAE": MAE,
    "MSE": MSE,
    "RMSE": RMSE,
    "avg_precision": avg_precision,
    "avg_recall": avg_recall,
    "avg_F1_score": avg_F1_score,
    "avg_exec": avg_exec
}

# Write the metrics to wandb.log format
wandb.log(metrics_dict)



#print('Precision: ', precision )
print('MAE: ', MAE )
#print('number of files: ', num_files)

with open(csv_file_aggregate, mode='a', newline='') as file_aggregate:
    writer_aggregate = csv.writer(file_aggregate)

    # Check if the file is empty
    file_empty = os.stat(csv_file_aggregate).st_size == 0

    # Write the header row only if the file is empty
    if file_empty:
        writer_aggregate.writerow(['model_name','Timestamp', 'confidence', 'param1' ,'number of prompts',  'Name', 'avg time performance','MAE','MSE' , 'RMSE', 'MAPE', 'FP Rate', 'FN Rate', 'overall_accuracy' , 'avg_overall_error' , 'avg_precision', 'avg_recall', 'avg_F1_Score','testing_iou_threshold'])

    # Write the average metrics to the CSV file
    writer_aggregate.writerow([model_name]+[timestamp, param2 , param1, prompt_number] + [data_path.replace("/", "_")] + [avg_exec, MAE , MSE, RMSE, MAPE , FP_rate , FN_rate , overall_accuracy, avg_overall_error , avg_precision, avg_recall, avg_F1_score, test_iou_threshold])



    




