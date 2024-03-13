# 3D localisation from multi-view X-ray projections

## Overview
This project, conducted in collaboration with SAFRAN R&T, aims to develop a geometry-based model for detecting 3D indications from X-ray multi-view images. By leveraging the LIDC-IDRI dataset, which is a valuable resource for advancing research in lung cancer detection, our objective is to implement state-of-the-art techniques in computer vision and deep learning to accurately identify indications in X-ray images.

## Dataset
To get started with the project, you'll need to download the LIDC-IDRI dataset from the following link: [LIDC-IDRI Dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254). This dataset contains a comprehensive collection of lung CT scans, annotated by expert radiologists, which serve as the ground truth for our detection task.

## Methodology
Our approach to detecting 3D indications from X-ray images is inspired by recent advancements in deep learning and geometric modeling. We draw upon techniques proposed in the paper "3D Bounding Box Estimation Using Deep Learning and Geometry" to guide our methodology. Here's an outline of our approach:

1. **Orientation estimation**: We use a novel MultiBin module to estimate the orientation of indications in the X-ray images. This module, based on deep learning techniques, helps in determining the spatial orientation of the detected objects.

2. **Dimension estimation**: A fully connected network is employed to estimate the dimensions of the detected indications. By leveraging deep learning models, we aim to accurately predict the size and scale of the identified objects.

3. **Geometric constraint**: To refine our detections, we apply geometric constraints based on projective geometry principles. This step helps in aligning the detected indications with the underlying geometry of the X-ray images.


#### We also implemented a Faster R-CNN fine-tuning approach to tackle this challenge. Please read the Project Report for more information.
