import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

from importlib import reload
import torch

## library to read lung cancer database
from sqlalchemy import func # required to query the db
import thirdparty.pylidc.pylidc as pl

from diffdrr.drr import DRR
from diffdrr.data import load_example_ct
from diffdrr.visualization import plot_drr


from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
import pydicom
import pylab
from scipy import ndimage
import skimage.io as io
import skimage.filters as flt
import scipy.ndimage.filters as flt
import warnings


def bbox_to_3d_coord(bbox):
    ''' Using this function to get the coordinates of the bbox in 3D. '''
    
    x_min, x_max = bbox[0].start, bbox[0].stop
    y_min, y_max = bbox[1].start, bbox[1].stop
    z_min, z_max = bbox[2].start, bbox[2].stop

    dico = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'z_min': z_min,
        'z_max': z_max
    }
    
    return dico


def crop_white_surrounding(image_path):
    """ Get the shadow of the 3D bbox. """
    
    image = Image.open(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    # Find the contours of the white box
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask to remove the white surroundings
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def find_bounding_box_coordinates(image_path):
    """ Find the bounding box coordinates of the white region in a binary or grayscale image. """
    
    image = Image.open(image_path)
    image_np = np.array(image)

    # Convert the image to grayscale if needed
    if len(image_np.shape) == 3:
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_np


    if image_gray.dtype != np.uint8:
        image_gray = np.uint8(image_gray)

    # Find corners
    corners = cv2.goodFeaturesToTrack(image_gray, 2000, 0.01, 5)

    if corners is None:
        return ([[]], 0)

    xylist = [(int(c[0][0]), int(c[0][1])) for c in corners]
    
    res = [xylist[i:i+4] for i in range(0, len(xylist), 4)]  # Generate a list of lists for rectangles
            
    return res, len(res)  # Return the result (list of lists with the coordinates) and the number of rectangles detected

image_path = "C:/Users/pcc/Desktop/Pole_IA_S7/rxtools-main/Example_annotations/3D_annotations/ann_2.png"
# find_bounding_box_coordinates(image_path)



#### Creating the image/annotation table

def create_label_file(patient_id, idx_x, bbox_info, path):
    filename = f"{10*(patient_id-1)+idx_x:06d}.txt"
    file_path = os.path.join(path, filename)
    
    with open(file_path, 'w') as label_file:
        label_file.write(" ".join(map(str, bbox_info)) + '\n')

def data_preprocessing(patient_id):
    ann = pl.query(pl.Annotation).filter(pl.Annotation.scan.has(pl.Scan.patient_id == f"LIDC-IDRI-{patient_id:04d}")).first()
    
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == f"LIDC-IDRI-{patient_id:04d}").first()
    volume = scan.to_volume()
    volume = volume.astype(np.float32)

    
    ##### Geometry acquisition :
    spacing = np.array([scan.pixel_spacing,scan.pixel_spacing, scan.slice_spacing])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rot_x=0.0
    rot_y=0.0
    rot_z=3*torch.pi/2
    rotation = torch.tensor([[rot_x, rot_y, rot_z]], device=device)

    bx, by, bz = torch.tensor(vol_ann.shape, dtype=torch.double) * \
                 torch.tensor(spacing, dtype=torch.double) / 2

    translation = torch.tensor([[bx, by, bz]], device=device) 
    
    #### Annotations 
    vol_ann = np.zeros(volume.shape)
    x_min, x_max, y_min, y_max, z_min, z_max = 0, 0, 0, 0, 0, 0
    
    if ann:
        vol_ann[ann.bbox()] = 1  
        coord = bbox_to_3d_coord(ann.bbox())
        x_min, x_max = coord['x_min'], coord['x_max']
        y_min, y_max = coord['y_min'], coord['y_max']
        z_min, z_max = coord['z_min'], coord['z_max']


    drr = DRR(
        volume,
        spacing,     
        sdr=300.0,   
        height=200,  
        delx=4.0,
    ).to(device)

    drr_ann = DRR(
        vol_ann,      
        spacing,     
        sdr=300.0,
        height=200,  
        delx=4.0,
    ).to(device)

    translation = torch.tensor([[bx.item(), by.item(), bz.item()]], dtype=torch.double, device=device)

    path = "C:/Users/pcc/Desktop/data_project/clean_imgs/"
    path_ann = "C:/Users/pcc/Desktop/data_project/ann_imgs/"
    path_labels = "C:/Users/pcc/Desktop/data_project/labels/"
    
    N_views = 10
    for idx_x in range(N_views):
        rad_x = idx_x * 2 * torch.pi / N_views
        rotation = torch.tensor([[rad_x, 0.0, 3 * torch.pi / 2]],
                            dtype=torch.double, device=device)

        img = drr(rotation, translation,
                        parameterization="euler_angles", convention="ZYX")
        
        img_ann = drr_ann(rotation, translation,
                        parameterization="euler_angles", convention="ZYX")


        filename = f"{10*(patient_id-1)+idx_x:06d}.png"
        # Save plots
        plot_drr(img_ann, ticks=False)
        plt.axis('off')
        plt.savefig(os.path.join(path_ann, filename), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        plot_drr(img, ticks=False)
        plt.axis('off')
        plt.savefig(os.path.join(path, filename), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Calculate bounding box information
        img_path = os.path.join(path_ann, filename)
        coordinates, num_rectangles = find_bounding_box_coordinates(img_path)
        image_class = 1 if num_rectangles > 0 else 0
        
        if image_class > 0:
            rectangle_coords = coordinates[0]    # I am considering only one nodule per pateint here
            x_min, y_min = min(rectangle_coords)
            x_max, y_max = max(rectangle_coords)
            
            if abs(x_max - x_min)<10:   # Making sure, we get a clear bounding box
                x_max = x_min + 10
                
            if abs(y_max - y_min)<10:
                y_max = y_min + 10
        else:
            x_min = y_min = x_max = y_max = 0


        dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
        
        if dx != 0 and dy != 0 :
            depth = (dx * scan.slice_thickness + (dx -1) * scan.slice_spacing)*0.00026458   # The last coefficient is to go from pixel to meters.
            width = (dy * scan.slice_thickness + (dy -1) * scan.slice_spacing)*0.00026458
            height = (dz * scan.slice_thickness + (dz -1) * scan.slice_spacing)*0.00026458
        
        else:
            depth = width = height = 0
        
        bbox_info = [image_class, rad_x, x_min, y_min, x_max, y_max, depth, width, height]

        # Save bounding box information to text file
        create_label_file(patient_id, idx_x, bbox_info, path_labels)


# patient_id = 1  
# data_preprocessing(patient_id)
