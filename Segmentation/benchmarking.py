import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment
from skimage.measure import label

def calculate_cell_metrics(gt_mask, pred_mask, iou_threshold=0.5):
    """
    Calculates True Positives, False Positives, False Negatives, 
    Precision, Recall, and F1-score for cell segmentation masks.
    """
    # Ensure masks are labeled integer arrays
    gt_labels = np.unique(gt_mask)[1:]  # skip background (0)
    pred_labels = np.unique(pred_mask)[1:]
    
    num_gt = len(gt_labels)
    num_pred = len(pred_labels)
    
    if num_gt == 0 and num_pred == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if num_gt == 0 or num_pred == 0:
        return 0.0, 0.0, 0.0, 0, num_pred, num_gt

    # Build an IoU matrix between all ground truth and predicted cells
    iou_matrix = np.zeros((num_gt, num_pred))
    
    for i, g_label in enumerate(gt_labels):
        g_indices = (gt_mask == g_label)
        for j, p_label in enumerate(pred_labels):
            p_indices = (pred_mask == p_label)
            
            # Intersection and Union calculation
            intersection = np.logical_and(g_indices, p_indices).sum()
            union = np.logical_or(g_indices, p_indices).sum()
            
            if union > 0:
                iou_matrix[i, j] = intersection / union

    # Use Hungarian matching to find optimal pairs based on IoU
    # We turn IoU into a cost matrix (1 - IoU) for minimization
    cost_matrix = 1.0 - iou_matrix
    gt_inds, pred_inds = linear_sum_assignment(cost_matrix)
    
    # Count True Positives based on the IoU threshold
    tp = 0
    for i, j in zip(gt_inds, pred_inds):
        if iou_matrix[i, j] >= iou_threshold:
            tp += 1
            
    fp = num_pred - tp
    fn = num_gt - tp
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, precision, recall, tp, fp, fn

# --- Example Usage ---
# Load your masks (assuming they are saved as label TIFFs)
ground_truth = tifffile.imread("path_to_your_ground_truth.tif")
watershed_pred = tifffile.imread("path_to_your/Segmentation.tif")

f1, prec, rec, tp, fp, fn = calculate_cell_metrics(ground_truth, watershed_pred, iou_threshold=0.5)

print(f"--- Watershed Benchmark Results ---")
print(f"True Positives (Correctly found cells): {tp}")
print(f"False Positives (False alarms): {fp}")
print(f"False Negatives (Missed cells): {fn}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-Score: {f1:.3f}")