README: Paper Homography Detection and Rectification

Author: Chanakya Nalapareddy, Nosherwan Babar  
Date: 12/11/2024  

---

PROGRAM DESCRIPTION:  
This program implements a comprehensive pipeline for detecting and rectifying a document from an input image using computer vision techniques. The key steps include:  
1. Preprocessing the image.  
2. Applying Gaussian smoothing.  
3. Performing edge detection using the Canny algorithm.  
4. Detecting lines using the Hough Transform.  
5. Finding intersections to identify the document corners.  
6. Computing a homography matrix for rectification.  
7. Generating a rectified (warped) version of the document.  

---

FEATURES:
1. All-in-One Script:
   - The entire code is contained in a single `.m` file for simplicity.
   - No additional files or dependencies are required.

2. Custom Implementations:
   - Implements key functions such as Hough Transform, Gaussian smoothing, and Rectification using customs functions.

3. Easy Setup:
   - Users only need to modify the file path for the input image to get started.
   - Generates a rectified version of the document using homography mapping.

---

INSTRUCTIONS TO RUN THE PROGRAM:

1. Requirements:
   - MATLAB (R2018b or later recommended).

2. Input Image:
   - Place your input image (e.g., `test_image.jpg`) in the same directory as the `.m` file.
   - Ensure the input image contains a clear view of the document.

3. Run the Script:
   - Open MATLAB and navigate to the directory containing the `.m` file.
   - Run the script:
     ```
     >> run('your_script_name.m')
     ```
   - Replace `your_script_name.m` with the name of the provided `.m` file.

4. Modify File Path:
   - Update the input image file path at the beginning of the script if required:
     ```matlab
     img = imread('your_image_file.jpg'); % Replace with your file name
     ```

5. Outputs:
   - The script generates and displays intermediate results:
     - Grayscale image.
     - Edge-detected image.
     - Hough Transform accumulator.
     - Lines superimposed on the edge-detected image.
     - Detected corners on the original image.
     - Rectified document image.
   - Saves the following files in the same directory:
     - `resized_image.jpg`: Resized grayscale input image.
     - `final_filtered_detected_image.png`: Final image with corners marked.
     - `rectified_image.jpg`: Rectified document image.

---

NOTES:
1. For best results, ensure the input image has sufficient resolution and the document edges are well-defined.
2. If no document is detected, try adjusting the Canny thresholds or Hough peak parameters directly in the script.

---
