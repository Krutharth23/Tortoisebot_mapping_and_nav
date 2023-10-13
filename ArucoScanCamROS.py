#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def work():

    class ArucoMarkerDetector:

        def __init__(self):
            self.cv_bridge = CvBridge()
            self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
            #self.image_sub = rospy.Subscriber("/webcam/image_raw", Image, self.image_callback)

        def image_callback(self, msg):
            try:
                cv_frame = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                
                gray_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)
                
                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                aruco_params = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
                corners, ids, _ = detector.detectMarkers(gray_frame)
                
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(cv_frame, corners, ids)
                    
                    for i in range(len(ids)):
                        # Calculate the centroid of the marker's corners
                        corners = corners[i][0]
                        corners = corners.reshape(4, 2)
                        corners = corners.astype(int)
                        top_right = corners[0].ravel()
                        top_left = corners[1].ravel()
                        bottom_right = corners[2].ravel()
                        bottom_left = corners[3].ravel()

                        centroid_x = np.mean([top_right[0], top_left[0], bottom_right[0], bottom_left[0]]).astype(int)
                        centroid_y = np.mean([top_right[1], top_left[1], bottom_right[1], bottom_left[1]]).astype(int)
                        centroid = (centroid_x, centroid_y)
                        
                        # Draw centroid
                        cv2.circle(cv_frame, centroid, 5, (0, 0, 255), -1)
                        cv2.imshow("frame", cv_frame)
                        cv2.waitKey(1)
                        print(f"Marker ID: {ids[i][0]}, Centroid: {centroid}")
                    return 1
                
                cv2.imshow("frame", cv_frame)
                cv2.waitKey(1)
            
            except Exception as e:
                print("Exception:", e)


    
    rospy.init_node("aruco_marker_detector")
    node = ArucoMarkerDetector()
    rospy.spin()

work()
