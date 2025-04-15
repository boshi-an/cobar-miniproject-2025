from flygym.vision import Retina
import cv2
import numpy as np
import time
from collections import deque

class VisualNavigator :
    
    def __init__(
        self,
        detection_zone_height=54,
        detection_zone_width=80,
        monitor_zone_height=64,
        monitor_zone_width=80,
        bottom_compensation=30,
        history_length=3,
        ) :
        
        self.retina = Retina()
        self.last_result = None
        self.doging = False
        
        self.detection_zone_height = detection_zone_height
        self.detection_zone_width = detection_zone_width
        self.monitor_zone_height = monitor_zone_height
        self.monitor_zone_width = monitor_zone_width
        self.bottom_compensation = bottom_compensation
        
        self.history_detection_zone = deque(maxlen=history_length)
        self.history_monitor_zone = deque(maxlen=history_length)
        self.no_obstacle_steps = 0
        self.no_monitor_steps = 0
        self.history_length = history_length
        self.last_obstacle_position = None
 
    def get_human_readable(self, hex_pxls) :
        
        return [
            255 - 255 * self.retina.hex_pxls_to_human_readable(
                hex_pxls[eye], True
            ).max(axis=2)[::2, ::2].astype(np.uint8)
            for eye in range(2)
        ]
    
    def visualization(self, left_img, right_img, last_obstacle_position, doging) :
        
        show_left = left_img.copy()
        show_right = right_img.copy()
        show_left[:self.detection_zone_height, -self.detection_zone_width:] = 128
        show_left[-self.detection_zone_height-self.bottom_compensation:, -self.detection_zone_width:] = 128
        show_left[:self.monitor_zone_height, -self.monitor_zone_width-self.detection_zone_width:-self.detection_zone_width] = 64
        show_left[-self.monitor_zone_height-self.bottom_compensation:, -self.monitor_zone_width-self.detection_zone_width:-self.detection_zone_width] = 64
        show_right[:self.detection_zone_height, :self.detection_zone_width] = 128
        show_right[-self.detection_zone_height-self.bottom_compensation:, :self.detection_zone_width] = 128
        show_right[:self.monitor_zone_height, self.detection_zone_width:self.monitor_zone_width+self.detection_zone_width] = 64
        show_right[-self.monitor_zone_height-self.bottom_compensation:, self.detection_zone_width:self.monitor_zone_width+self.detection_zone_width] = 64
        show_img = np.concatenate((show_left, show_right), axis=1)
        
        if last_obstacle_position is not None :
            # draw the obstacle position
            cv2.circle(show_img, (left_img.shape[1] - self.detection_zone_width + int(last_obstacle_position), int(self.detection_zone_height)), 10, (0, 0, 255), -1)
        
        if doging :
            # Write "Doiging" on the image
            cv2.putText(show_img, "Doiging", (left_img.shape[1], int(self.detection_zone_height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Image", show_img)
        cv2.waitKey(1)

    def get_obstacle_pos(self, hex_pxls, updated) :
        
        if updated :
            left, right = self.get_human_readable(hex_pxls)
            
            # display filtered_left
            left_detection_zone_1 = left[:self.detection_zone_height, -self.detection_zone_width:]
            left_detection_zone_2 = left[-self.detection_zone_height-self.bottom_compensation:, -self.detection_zone_width:]
            
            right_detection_zone_1 = right[:self.detection_zone_height, :self.detection_zone_width]
            right_detection_zone_2 = right[-self.detection_zone_height-self.bottom_compensation:, :self.detection_zone_width]
            
            left_monitor_zone_1 = left[:self.monitor_zone_height, -self.monitor_zone_width-self.detection_zone_width:]
            left_monitor_zone_2 = left[-self.monitor_zone_height-self.bottom_compensation:, -self.monitor_zone_width-self.detection_zone_width:]
            
            right_monitor_zone_1 = right[:self.monitor_zone_height, :self.monitor_zone_width+self.detection_zone_width]
            right_monitor_zone_2 = right[-self.monitor_zone_height-self.bottom_compensation:, :self.monitor_zone_width+self.detection_zone_width]
            
            detection_zone_1 = np.concatenate((left_detection_zone_1, right_detection_zone_1), axis=1)
            detection_zone_2 = np.concatenate((left_detection_zone_2, right_detection_zone_2), axis=1)
            detection_zone = np.concatenate((detection_zone_1, detection_zone_2), axis=0)
            self.history_detection_zone.append(detection_zone)
            
            monitor_zone_1 = np.concatenate((left_monitor_zone_1, right_monitor_zone_1), axis=1)
            monitor_zone_2 = np.concatenate((left_monitor_zone_2, right_monitor_zone_2), axis=1)
            monitor_zone = np.concatenate((monitor_zone_1, monitor_zone_2), axis=0)
            self.history_monitor_zone.append(monitor_zone)
            
            monitor_zone_has_obstacle = (monitor_zone < 250).any()
            
            # get the mean of second axis of where obstacle
            # obstacle_positions = np.where(np.min(self.history_detection_zone, axis=0) < 250)
            obstacle_positions = np.where(detection_zone < 250)
            obstacle_mean_position = np.mean(obstacle_positions[1]) if obstacle_positions[1].size > 0 else None
            
            if obstacle_mean_position is not None :
                self.no_obstacle_steps = 0
                self.last_obstacle_position = obstacle_mean_position
            else :
                self.no_obstacle_steps += 1
                if self.no_obstacle_steps > self.history_length :
                    self.last_obstacle_position = None
            
            if self.last_obstacle_position is None :
                if self.doging and monitor_zone_has_obstacle :
                    result = (1, 1)
                    self.no_monitor_steps = 0
                elif self.doging and self.no_monitor_steps < self.history_length and not monitor_zone_has_obstacle :
                    result = (1, 1)
                    self.no_monitor_steps += 1
                elif self.doging and self.no_monitor_steps >= self.history_length and not monitor_zone_has_obstacle :
                    result = (0, 0)
                    self.doging = False
                else :
                    result = (0, 0)
                    self.doging = False
            elif self.last_obstacle_position < self.detection_zone_width :
                result = (1, -1)
                self.doging = True
            elif self.last_obstacle_position >= self.detection_zone_width :
                result = (-1, 1)
                self.doging = True
            else :
                raise ValueError("mean_position is strange")
            self.last_result = result
            self.visualization(left, right, self.last_obstacle_position, self.doging)
            return result
        else :
            assert self.last_result is not None, "No previous result available"
            return self.last_result