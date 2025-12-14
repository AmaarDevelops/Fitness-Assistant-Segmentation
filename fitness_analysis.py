import numpy as np
import cv2
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

def calculate_centroid(mask):

    M = cv2.moments(mask)

    if M["m00"] == 0:
        return None,None

    # Centroid coordinates are calculated from the moments:
    # Cx = m10 / m00 , describes how the area of mask is spread horizontally
    # Cy = m01 / m00 , describes how the area of mask is spread vertically
    Cx = int(M['m10']) / int(M['m00'])
    Cy = int(M['m01']) / int(M['m00'])

    return Cx,Cy




# ---------- Calculate orientation ------

def calculate_orientation(mask):
    M = cv2.moments(mask)

    if M["m00"] == 0: # m00 the total area / pixel intensities of the entire mask
        return None

    mu_20 = M['mu20']
    mu_02 = M['mu02']
    mu_11 = M['mu11']

    # Formula for Orientation (theta) based on central moments:
    # theta = 0.5 * arctan2( 2 * mu_11, (mu_20 - mu_02) )

    angle_rad = 0.5 * np.arctan2(2*mu_11,mu_20 - mu_02)

    angle_deg = np.degrees(angle_rad) # Converting into degrees from radians

    if angle_deg < 0: # Adjust the angle to be intuitive b/w 0 and 180
        angle_deg += 180

    return angle_deg



# ---- Analyse mask metric ----


def analyse_metric_mask(mask):
    cX,cY = calculate_centroid(mask)
    angle = calculate_orientation(mask)

    if cX is None:
        return {"Centroid_X" : None , "Centriod_Y" : None, "Angle_degrees" : None}

    return {
        "Centroid_X" : cX,
        "Centroid_Y" : cY,
        "Angle_Degrees" : angle
    }


# Real-time Form correction
def form_correction(angle_deg):
    if angle_deg is None:
        return None, (0,0,0) # Black text

    MIN_ANGLE = 70.0
    MAX_ANGLE = 110.0

    if angle_deg <= MIN_ANGLE or angle_deg >= MAX_ANGLE:
        return 'TILT WARNING' , (0,0,255)
    else:
        return 'FORM Ok' , (0,255,0)



# ----------- Running live analysis ---------

def run_live_analysis(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print('Error webcam could not be opened')
        return

    yolo = YOLO('yolov8n-seg.pt')


    path_history = []

    while True:
        ret,frame = cap.read()

        if not ret:
            print('Failed to grab frame')
            break

        # Calculate YOLO predictions

        results = yolo(frame,stream=False,verbose=False,conf=0.5)

        # Calculate current mask
        current_mask = np.zeros(frame.shape[:2],dtype=np.uint8)


        # If an object is detected
        if results and results[0].masks is not None:
            # convert the mask into numpy array
            mask = results[0].masks.data.cpu().numpy()


            largest_mask_area = 0
            best_mask = None

            # Calculate the best mask and area
            for mask_tensor in mask:
                mask = cv2.resize(mask_tensor,(frame.shape[1],frame.shape[0]))
                area = np.sum(mask)

                if area > largest_mask_area:
                    largest_mask_area = area
                    best_mask = mask * 255


            if best_mask is not None:
                current_mask = best_mask.astype(np.uint8) # Convert best mask into numerical values (0-255)

                metrics = analyse_metric_mask(current_mask) # Analyse the metrics for the current (best) mask

                mask_overlay = cv2.merge([current_mask,current_mask,current_mask])
                frame = cv2.addWeighted(frame,1,mask_overlay,0.3,0)

            else:
                metrics = {"Centroid_X" : None, "Centroid_Y" : None, "Angle_Degrees" : None}
        else:
            metrics = {"Centroid_X" : None, "Centroid_Y" : None, "Angle_Degrees" : None}


        # Draw the green circle (dot)
        if metrics['Centroid_X'] is not None:
            cX,cY = int(metrics['Centroid_X']) , int(metrics['Centroid_Y'])
            angle_deg = metrics['Angle_Degrees']

            path_history.append((cX,cY))

            cv2.circle(frame,(cX,cY),5,(0,255,0),-1)

            # Draw the red line

            if angle_deg is not None:
                length = 150
                angle_rad = np.radians(angle_deg)
                x_end = int(cX + length * np.cos(angle_rad))
                y_end = int(cY + length * np.sin(angle_rad))

                cv2.line(frame,(cX,cY),(x_end,y_end),(0,0,255),2)

            form_text,form_color = form_correction(angle_deg)


            cv2.putText(frame,f"Angle : {angle_deg:.2f}" , (10,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            cv2.putText(frame,form_text,(10,80),cv2.FONT_HERSHEY_SIMPLEX,2,form_color,3)

        cv2.imshow('Live fitness analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if path_history:
        x_coords,y_coords = zip(*path_history)

        plt.figure(figsize=(10,9))

        plt.plot(x_coords,y_coords,label='Lift Path',color='red',linewidth=2)

        plt.scatter(x_coords[0],y_coords[0],color='green',marker='o',s=100)
        plt.scatter(x_coords[-1],y_coords[-1],color='blue',marker='X',s=100)

        plt.gca().invert_yaxis()

        plt.title('Centroid path tracking analysis')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()




if __name__ == "__main__":
    run_live_analysis()








