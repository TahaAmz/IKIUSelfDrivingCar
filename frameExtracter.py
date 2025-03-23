import os
import cv2

input_dir = r'E:\Dataset\Video'
output_dir = r'E:\Dataset\Frame'

video_files = [f for f in os.listdir(input_dir)]

for video_name in video_files:
    video_path = os.path.join(input_dir, video_name)
    vid = cv2.VideoCapture(video_path)
    
    if not vid.isOpened():
        print(f"failed to open {video_name}")
        continue
    
    current_frame = 0
    sign_name = os.path.splitext(video_name)[0]
    data_folder = os.path.join(output_dir, sign_name)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    while (True):

        success, frame = vid.read()
        if not success:
            break
        
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        cv2.imshow("output", rotated_frame)
        cv2.imwrite(os.path.join(data_folder, f"frame_{current_frame}.png"),
                                rotated_frame)
        current_frame += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    print(f"Completed:  {video_name} - {current_frame}")
    cv2.destroyAllWindows()