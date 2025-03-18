import cv2
import os

vid = cv2.VideoCapture(r'D:\Dataset\barred_area_raw.mp4')
current_frame = 0

if not os.path.exists('data'):
    os.makedirs('data')

while (True):

    success, frame = vid.read()
    cv2.imshow("output", frame)
    cv2.imwrite(r'D:/Dataset/frame' + str(current_frame) + ".png", frame)
    current_frame += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()