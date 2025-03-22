import os
import cv2 as cv
import numpy as np

# Define the folder containing the 962 frames and the template path
frames_folder = r'E:\Dataset\Test\barred_area_raw'   # folder with frames
template_path = r'E:\Dataset\Test\barred_area_raw_template.png'  # your single template image

# Define an output folder for annotated frames and text files
output_folder = r'E:\Dataset\Test\annotated_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all PNG files in the frames folder
frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.png')]
frame_files.sort()  # sort in case order matters

# Load the single template image in grayscale
template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
if template is None:
    raise Exception("Template image not found at: " + template_path)
template_w = template.shape[1]
template_h = template.shape[0]

# Set a matching threshold
threshold = 0.5

# Loop over each frame in the folder
for frame_file in frame_files:
    frame_path = os.path.join(frames_folder, frame_file)
    main_img = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
    if main_img is None:
        print(f"Skipping {frame_file} (unable to load image)")
        continue

    # Apply template matching
    result = cv.matchTemplate(main_img, template, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    if max_val >= threshold:
        print(f"Found object in {frame_file}")
        # Calculate bounding box coordinates
        top_left = max_loc
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        
        # Draw bounding box (draw on a copy so main_img is preserved if needed)
        bound_img = main_img.copy()
        cv.rectangle(bound_img, top_left, bottom_right, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
        
        # Save the annotated image to the output folder
        annotated_img_path = os.path.join(output_folder, frame_file)
        cv.imwrite(annotated_img_path, bound_img)
        
        # Calculate YOLO-style normalized coordinates
        x_center = (top_left[0] + bottom_right[0]) / 2.0
        y_center = (top_left[1] + bottom_right[1]) / 2.0
        x_center /= main_img.shape[1]
        y_center /= main_img.shape[0]
        width = template_w / main_img.shape[1]
        height = template_h / main_img.shape[0]
        
        # Create annotation text (assuming class id 0)
        annotation_text = f"0 {x_center} {y_center} {width} {height}"
        # Save the annotation in a text file with the same base name as the frame image
        txt_file = os.path.splitext(frame_file)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_file)
        with open(txt_path, 'w') as f:
            f.write(annotation_text)
    else:
        print(f"Object not found in {frame_file}")

# Wait for a key press and close all OpenCV windows (if any are opened)
cv.waitKey(0)
cv.destroyAllWindows()
