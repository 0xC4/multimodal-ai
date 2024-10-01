"""
Various methods for binarizing predictions to predictions + confidence.
"""

from typing import Dict, Tuple
import numpy as np
from scipy import ndimage

def _enforce_deadzone(arr, outer_deadzone_pixels: int = 3, value=0.):
    """
    Sets outer border of X pixels to a given value.
    """
    dz = outer_deadzone_pixels
    if dz < 1:
        return arr
    mask = np.zeros_like(arr, dtype=int)
    mask[dz:-dz, dz:-dz, dz:-dz] = 1
        
    return arr * mask

def static_threshold(
    predictions: np.ndarray,
    threshold: float = 0.10,
    min_voxels: int = 25,
    outer_deadzone: int = 3
    ) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Binarizes raw predictions with given threshold, then performs connected-
        components to get separate detections, and filters them by a minimum 
        size in voxels.

    Returns: 
        label_img: Numpy array with unique values for each separate detection
            in the image.
        confidences: Dictionary linking each detection with its confidence value
            determined as the maximum prediction in the detection.
        outer_deadzone: Number of pixels on the edge of the array where 
            detections are disallowed. Prevents FP detections on the edge due
            to padding in conv layers.
    """

    # Load and Preprocess predictions Image
    confidences = {}
    predictions = predictions.copy().round(3)
    if outer_deadzone > 0:
        predictions = _enforce_deadzone(predictions, outer_deadzone)
    
    binary_pred = (predictions >= threshold) * 1.
    label_img, num_detections = ndimage.label(binary_pred, np.ones((3, 3, 3)))
    
    # Create a new label image, with lesion labels from 1-N, instead of having 
    # intermediate indexes unused due to filtering based on volume.
    out_label_img = np.zeros_like(label_img)

    # Loop over all detections in the binarized heatmap
    out_det_idx = 1
    for detection_idx in range(1, num_detections+1):

        # determine mask for current lesion
        binarized_detection = (label_img == detection_idx) * 1.

        # Remove detections smaller than X voxels
        if np.sum(binarized_detection) <= min_voxels:
            continue

        # Add the lesion to the output label image
        out_label_img += (binarized_detection * out_det_idx).astype('int32')

        # Set the confidence as the maximum prediction for this detection
        confidences[out_det_idx] = np.amax(binarized_detection * predictions)
        out_det_idx += 1

    return out_label_img, confidences

def dynamic_threshold(
    prediction: np.ndarray,
    min_voxels: int = 25,
    outer_deadzone: int = 3,
    num_lesions_to_extract: int = 5,
    dynamic_threshold_factor: float = 0.75,
    remove_adjacent_detections: bool = True,
    minimum_confidence: float = 0.15
    ) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Get the top X most likely predictions by detecting the most likely lesion, 
        and subsequently removing it from the prediction, and repeating the 
        process. 
    Threshold is adjusted "dynamically" by halfing it compared to the current
        maximum in the image after each iterations.
    """

    out_confidences = {}
    masked_prediction = prediction.copy().round(3)
    if outer_deadzone > 0:
        masked_prediction = _enforce_deadzone(masked_prediction, outer_deadzone)
    out_label_img = np.zeros_like(prediction, dtype=int)
    
    # Create a new label image, with lesion labels from 1-N, instead of having 
    # intermediate indexes unused due to filtering based on volume.
    for out_det_idx in range(1, num_lesions_to_extract+1):
        # Determine the highest likelihood prediction in the heatmap
        max_confidence = np.amax(masked_prediction)
        threshold = max_confidence * dynamic_threshold_factor

        # If the highest predicted likelihood in the image is below the minimum
        if max_confidence < minimum_confidence: 
            break

        # Get all separate detections and confidences in the image 
        # Always set outer deadzone to 0, because we already did this in the 
        # beginning
        label_img, confidences = static_threshold(
            masked_prediction, threshold, min_voxels, outer_deadzone=0)

        # Obtain highest likelihood detection
        if not any(confidences):
            break

        max_confidence_lesion_idx = max(confidences, key = confidences.get) 
        mask_current_lesion = (label_img == max_confidence_lesion_idx) * 1.

        # Remove lesion from masked prediction
        masked_prediction = (masked_prediction * (1 - mask_current_lesion))

        # Detect whether there is overlap with already extracted lesions.
        dilated_lesions = ndimage.binary_dilation(out_label_img > 0)
        has_overlap = np.sum(mask_current_lesion * dilated_lesions) > 0

        # If there is overlap, skip
        if has_overlap and remove_adjacent_detections:
            continue

        # Otherwise store the extracted lesion
        out_confidences[out_det_idx] = max_confidence
        out_label_img += (mask_current_lesion * out_det_idx).astype('int32')

    return out_label_img, out_confidences
