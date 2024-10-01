from glob import glob
from os import path

import numpy as np
import SimpleITK as sitk

from .helpers import dynamic_threshold

# The directory where your heatmaps are stored
PREDICTIONS_PATH = "./heatmaps/"

# The output file to which your data on individual lesions will be written
LESIONS_OUTPUT_CSV = "./lesions.csv"

# Create / empty the lesion output CSV file
with open(LESIONS_OUTPUT_CSV, "w+") as f:
    pass

# Iterate over heatmaps
for sample_idx, heatmap_path in enumerate(sorted(glob(PREDICTIONS_PATH + "*.nii.gz"))):
    print("Processing", heatmap_path)
    # Read the image and extract the numpy array
    heatmap_s = sitk.ReadImage(heatmap_path, sitk.sitkFloat32)
    heatmap_n = sitk.GetArrayFromImage(heatmap_s).T

    # Extract the patient ID and visit date from file name formatted as:
    # /path/to/file/[patient_id]_[visit_date].nii.gz
    visit_id = path.basename(heatmap_path).split(".")[0]
    patient_id = visit_id.split("_")[0]
    visit_date = visit_id.split("_")[1]
    print("Patient ID:", patient_id)
    print("Date:", visit_date)

    # Extract the spacing so that we can calculate the volume of a single voxel
    spacing = heatmap_s.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    # Apply a dynamic threshold to extract lesions from detection heatmap
    blobs, confs = dynamic_threshold(heatmap_n)

    # Process each lesion and obtain its likelihood score and volume
    print(f"Detected {len(confs)} lesions:")
    lesions = []
    for lesion_idx in confs:
        print(lesion_idx, confs[lesion_idx])
        lesion_size = np.sum((blobs==lesion_idx)*1)*voxel_volume
        lesions.append((confs[lesion_idx], lesion_size))

    # If fewer than three lesions are detected, append 0's 
    lesions += [(0,0)] * 5
    lesions = sorted(lesions, reverse=True)

    lesion_0_lkhd = lesions[0][0]
    lesion_0_size = lesions[0][1]
    lesion_1_lkhd = lesions[1][0]
    lesion_1_size = lesions[1][1]
    lesion_2_lkhd = lesions[2][0]
    lesion_2_size = lesions[2][1]

    print("Extracted data:", lesions)

    # Append a new row in the output CSV
    with open(LESIONS_OUTPUT_CSV, "a") as f:
        f.write(f"{patient_id};{visit_date};{lesion_0_lkhd};{lesion_0_size};{lesion_1_lkhd};{lesion_1_size};{lesion_2_lkhd};{lesion_2_size}\n")

print("Done.")