import argparse
import cv2
from pupil_apriltags import Detector
import numpy as np
import time

class roadSigns:
    def __init__(self):
        # Initialize the camera and arguments
        self.args = self.get_args()
        self.cap = cv2.VideoCapture(self.args.camera)               # "http://192.168.x.x:4747/video" for DroidCam, (self.args.camera) for default=1
        
        if not self.cap.isOpened():
            print("Error: camera could not be opened!")
            exit()
    
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)     # Video width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)   # Video height
        self.cap.set(cv2.CAP_PROP_FPS, self.args.fps)               # Video FPS
        
        # Sign detector using Pupil_Apriltags
        self.sign_detector = Detector(
            families=self.args.family,
            nthreads=self.args.nthread,
            quad_decimate=self.args.quad_decimate,
            quad_sigma=self.args.quad_sigma,
            refine_edges=self.args.refine_edges,
            decode_sharpening=self.args.decode_sharpening,
            debug=self.args.debug
            )
        
        # Signs, their Apriltag ids and tasks related to each sign
        self.sign_names = {
            0 : ("Tunnel beginning", self.tunnel_beginning_task),
            1 : ("Tunnel end", self.tunnel_end_task),
            2 : ("Cross walk", self.cross_walk_task),
            3 : ("Parking zone", self.parking_zone_task),
            4 : ("No passing zone beginning", self.no_passing_zone_beginning_task),
            5 : ("No passing zone end", self.no_passing_zone_end_task),
            6 : ("Stop", self.stop_task),
            7 : ("Priority over", self.priority_over_task),
            8 : ("Barred area", self.barred_area_task),
            9 : ("Steep hill uphill", self.steep_hill_uphill_task),
            10 : ("Steep hill downhill", self.steep_hill_downhill_task),
            11 : ("Turn left", self.turn_left_task),
            12 : ("Turn right", self.turn_right_task),
            119 : ("Go straight", self.go_straight_task)
        }
        
        # Track the last recognized sign
        self.sign_last_processed_time = {}
        self.cooldown_period = 1
    
    # Argument parser
    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--camera", type=int, default=1) # webcam is default=1
        parser.add_argument("--width", type=int, default="640")
        parser.add_argument("--height", type=int, default="480")
        parser.add_argument("--fps", type=int, default=30)
        parser.add_argument("--family", type=str, default="tag36h11")
        parser.add_argument("--nthread", type=int, default=1)
        parser.add_argument("--quad_decimate", type=float, default=2.0)
        parser.add_argument("--quad_sigma", type=float, default=0.0)
        parser.add_argument("--refine_edges", type=int, default=1)
        parser.add_argument("--decode_sharpening", type=float, default=0.25)
        parser.add_argument("--debug", type=int, default=0)
        return parser.parse_args()
    
    # Assign tasks to signs
    def sign_handle(self, sign_id):
        current_time = time.time()
        if (sign_id not in self.sign_last_processed_time or
            current_time - self.sign_last_processed_time[sign_id] > self.cooldown_period):
            
                self.sign_last_processed_time[sign_id] = current_time
                sign_name, task = self.sign_names.get(sign_id, ("Unknown sign!", lambda: None))
                # print(f"Detected: {sign_name}")
                task()
            
        
    # Sort signs by size and show sign names in the video
    def draw_signs(self, img, tags):
        tags_sorted = sorted(tags, key=lambda t: (t.corners[0][1] - t.corners[0][0]) * (t.corners[2][1] - t.corners[0][1]), reverse=True)
        
        for tag in tags_sorted:
            corners = [tuple(map(int, c)) for c in tag.corners]
            center = tuple(map(int, tag.center))
            cv2.circle(img, center, 5, (255, 0, 0), 2)
            cv2.polylines(img, [np.array(corners)], True, (255, 0, 0), 2)
            
            sign_id = tag.tag_id
            sign_name = self.sign_names.get(sign_id, ("Unknown sign!", None))[0]
            cv2.putText(img, f"{sign_name} ({sign_id})", (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # print(f"tag id: {sign_id}, sign name: {sign_name}")
            self.sign_handle(sign_id)
        return img
    
    ### Tasks assined to each sign ###
    
    def tunnel_beginning_task(self):
        print("Tunnel beginning")

    def tunnel_end_task(self):
        print("Tunnel end")

    def cross_walk_task(self):
        print("Cross walk")

    def parking_zone_task(self):
        print("Parking zone")

    def no_passing_zone_beginning_task(self):
        print("No passing zone beginning")
        
    def no_passing_zone_end_task(self):
        print("No passing zone end")

    def stop_task(self):
        print("Stop")

    def priority_over_task(self):
        print("Priority over")

    def barred_area_task(self):
        print("Barred area")

    def steep_hill_uphill_task(self):
        print("Steep hill uphill")
        
    def steep_hill_downhill_task(self):
        print("Steep hill downhill")

    def turn_left_task(self):
        print("Turn left")

    def turn_right_task(self):
        print("Turn right")

    def go_straight_task(self):
        print("Go straight")
        
    # Sign detection
    def detect(self):
        while True:
            ret, image = self.cap.read()
            if not ret:
                print("Error: Failed to capture!")
                break
        
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = np.uint8(gray)
            tags = self.sign_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
            output_image = self.draw_signs(image, tags)

            cv2.imshow("Sign Detection", output_image)
            
            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        