#import libraries

import cv2
import mediapipe as mp
import numpy as np
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose


#Streamlit 

import streamlit as st
import pandas as pd
header = st.container()
user_input = st.container()
model = st.container()


with header:
    st.title('AI Gym Trainer')

with user_input:
    st.header('Enter your data')
    fn1 = st.text_input("Enter your name ")
choice =int(st.number_input('Excerise: 1.Bicep_curl 2.Shoulder_press 3.Highknees',max_value=3, step=1))
  


#calculate angles between joints

def calculate_angle(a,b,c):
    a = np.array(a)  #first angle
    b = np.array(b)  #second angle
    c = np.array(c)  #third angle
    
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle>180.0:
        angle = 360-angle
    return angle


#Curl Counter


cap = cv2.VideoCapture(0)

#creating curl counter variable
counter = 0
stage = None
error=0

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Selection of Excercise
        
        
        
        
        if(choice==1):
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle
                angle_1 =calculate_angle(left_shoulder, left_elbow, left_wrist)
                angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                # Visualize angle
                cv2.putText(image, str(angle_1), 
                               tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                cv2.putText(image, str(angle_2), 
                               tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                #CURL COUNTER LOGIC
                if angle_1 > 160 and angle_2 >160 :
                    stage = "Down"
                if angle_1 < 30 and angle_2 < 30 and stage == 'Down':
                    stage = "up"
                    counter +=1
                    print(counter)

            except:
                pass

            #setting up curl counter box
            cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

            #sending values to curl counter box
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            #printing hand stage while exercising

            cv2.putText(image, 'STAGE', (165,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (165,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            cv2.putText(image, 'DOUBLE BICEP CURLS', (390,18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            if counter >=1 and counter <2:
                        cv2.putText(image, '1 Set completed ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            if counter >=2 and counter <3:
                        cv2.putText(image, '2 Set completed ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            if counter ==3 :
                        cv2.putText(image, '3 Set completed ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            if counter >3 :
                        cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                        cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                     )
            
        elif(choice==2):
                try:
                    landmarks = results.pose_landmarks.landmark

                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calculate angle
                    angle_1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                    # Visualize angle
                    cv2.putText(image, str(angle_1), 
                                   tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    cv2.putText(image, str(angle_2), 
                                   tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    #CURL COUNTER LOGIC
                    if angle_1 > 110 and angle_2 >110 :
                        stage = "Up"
                    if angle_1 < 90 and angle_2 < 90 and stage == 'Up':
                        stage = "Down"
                        counter +=1
                        print(counter)


                except:
                    pass

                #setting up curl counter box
                cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                #sending values to curl counter box
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                #printing hand stage while exercising

                cv2.putText(image, 'STAGE', (165,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (165,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                cv2.putText(image, 'SHOULDER_PRESS', (390,18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=1 and counter <2:
                            cv2.putText(image, '1 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=2 and counter <3:
                            cv2.putText(image, '2 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter ==3 :
                            cv2.putText(image, '3 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >3 :

                            cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                            cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                         )            
        
        elif(choice==3):
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    # Calculate angle
                    angle = calculate_angle(shoulder, hip, knee)

                    # Visualize angle
                    cv2.putText(image, str(angle), 
                                   tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    #CURL COUNTER LOGIC
                    if angle > 160:
                        stage = "Down"
                    if angle < 80 and stage == 'Down':
                        stage = "Up"
                        counter +=1
                        print(counter)


                except:
                    pass

                #setting up curl counter box
                cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                #sending values to curl counter box
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                #printing hand stage while exercising

                cv2.putText(image, 'STAGE', (165,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (165,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                cv2.putText(image, 'HIGH KNEES', (390,18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=1 and counter <2:
                            cv2.putText(image, '1 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=2 and counter <3:
                            cv2.putText(image, '2 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter ==3 :
                            cv2.putText(image, '3 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >3 :

                            cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                            cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                         )     
        else:
            print("Invalid choice!!!")
            break
        
        cv2.imshow('VIDEO', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    

    cap.release()
    cv2.destroyAllWindows()