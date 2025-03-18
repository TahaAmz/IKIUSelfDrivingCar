#Auto labeling v3

import cv2 as cv
import numpy as np
import os

total_img_count = int(input("Enter the total imgs count: "))

pre_path = r'E:/frame'
b = 1

for a in range(total_img_count):
    raw_path = pre_path + str(b)
    b += 1
    main_img_path = raw_path + '.png'
    main_txt_path = raw_path + '.txt'

    # Load images
    main_img = cv.imread(main_img_path, cv.IMREAD_GRAYSCALE)
    
    # Match template

    val_list = []
    
    for c in range(1,6):
        
        result = cv.matchTemplate(main_img, cv.imread(r'C:\Users\asus\Desktop\template' + str(c) + '.png', cv.IMREAD_GRAYSCALE), cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        
        val_list.append(max_val)

    max_template_val = max(val_list)
    best_index = val_list.index(max_template_val)

    print(max_template_val)
    print(best_index)
    best_index += 1
    best_template = cv.imread(r'C:\Users\asus\Desktop\template' + str(best_index) + '.png', cv.IMREAD_GRAYSCALE)

    
    threshold = 0.5
    if max_val >= threshold:
        print('Found Object.')
        print('Best match top left position:', max_loc)
        print('Best match confidence:', max_val)
        
        obj_w = best_template.shape[1]
        obj_h = best_template.shape[0]

        top_left = max_loc
        bottom_right = (top_left[0] + obj_w, top_left[1] + obj_h)

        # Draw bounding box
        boundimg=cv.rectangle(main_img, top_left, bottom_right, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
        cv.imwrite(r'E:\check_frame' + str(b) + ".png", boundimg)
        # Calculate object position and size
        x_center = (top_left[0] + bottom_right[0]) / 2.0
        y_center = (top_left[1] + bottom_right[1]) / 2.0
        x_center /= main_img.shape[1]
        y_center /= main_img.shape[0]
        width = obj_w / main_img.shape[1]
        height = obj_h / main_img.shape[0]

        # Write detection information to text file
        final_text = '0 ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height)
        #print("check point for main txt path:", main_txt_path)
        with open(main_txt_path, 'w') as file:
            file.write(final_text)
    else:
        print('Object not found.')
        

    # Display the image with bounding box

cv.waitKey(0)
cv.destroyAllWindows()

