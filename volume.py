import os
import numpy as np
import pydicom
import cv2
import re
from vedo import Volume, show, settings

settings.use_depth_peeling = True

ct_folder = r"C:\Users\z005563b\Downloads\CT_CT"
#ct_folder = r"C:\Users\z005563b\Downloads\CT_folder"
mri_folder = r"C:\Users\z005563b\Downloads\CT_MR"
target_shape = (512, 512)

def get_slice_location_or_instance(dcm):
    """Return SliceLocation if available; otherwise InstanceNumber."""
    if hasattr(dcm, 'SliceLocation'):
        return float(dcm.SliceLocation)
    else:
        return 0  # fallback

def load_dicom_volume_sorted_by_slice_location(folder, target_shape=(512, 512)):
    dicoms = []
    for fname in os.listdir(folder):
        if fname.endswith(".dcm") or fname.endswith(".IMA"):
            path = os.path.join(folder, fname)
            try:
                dcm = pydicom.dcmread(path)
                location = get_slice_location_or_instance(dcm)
                dicoms.append((location, dcm))
            except Exception as e:
                print(f"Failed to read {path}: {e}")
            #print(f"Loaded {len(dicoms)} DICOM slices from: {folder}")

 
    # Sort based on slice location
    dicoms.sort(key=lambda x: x[0])
    
    volume = []
    for _, dcm in dicoms:
        img = dcm.pixel_array.astype(np.float32)
        img = cv2.resize(img, target_shape)
        volume.append(img)
    
    return np.array(volume)

def normalize(volume):
    volume -= volume.min()
    if volume.max() > 0:
        volume = volume / volume.max()
    return (volume * 255).astype(np.uint8)

# Load volumes sorted by SliceLocation
ct_volume = normalize(load_dicom_volume_sorted_by_slice_location(ct_folder, target_shape))
mri_volume = normalize(load_dicom_volume_sorted_by_slice_location(mri_folder, target_shape))

# Create vedo Volume objects
ct_vol = Volume(ct_volume).color('gray').alpha([0, 0.1, 0.9])
mri_vol = Volume(mri_volume).color('magenta').alpha([0, 0.2, 0.9])

# Fuse and display
fused = ((0.5 * ct_volume.astype(np.float32) + 0.5 * mri_volume.astype(np.float32))).astype(np.uint8)
fused_vol = Volume(fused).color('cyan').alpha([0, 0.2, 0.9])
show(fused_vol, "Fused CT + MRI Volume", axes=4, bg="black")

# Show CT and MRI volumes separately
#show(ct_vol, "CT Volume", axes=4, bg='black')
