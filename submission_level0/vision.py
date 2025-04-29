from flygym.vision import Retina
import cv2
import numpy as np
import time
from collections import deque
import torch
import os
from torch import nn

class VisualNavigator :
    
    def __init__(
        self,
        detection_zone_height=50,
        detection_zone_width=84,
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
    
    def _channels_to_rgb(self, channels) :

        rgb = np.zeros((channels.shape[0], channels.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 0] = channels[:, :, 0]
        rgb[:, :, 1] = channels[:, :, 1]

        return rgb
 
    def get_human_readable(self, hex_pxls) :
        
        return [
            255 * self._channels_to_rgb(self.retina.hex_pxls_to_human_readable(
                hex_pxls[eye], True
            )[::2, ::2].astype(np.uint8))
            for eye in range(2)
        ]

    def visualization(self, left_img, right_img, raw_vision, last_obstacle_position, doging) :
        
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
        
        cv2.imshow("Image", raw_vision[0])
        cv2.waitKey(1)

    def _get_raw_vision_mask(self, raw_vision) :

        mask_list = []
        mask_hex_list = []

        for eye in range(2) :
            mask = (raw_vision[eye, ..., 0:1] > raw_vision[eye, ..., 1:2])
            mask_hex = self.retina.raw_image_to_hex_pxls(
                raw_vision * mask
            )
            mask_hex_list.append(mask_hex)
            mask_list.append(mask)
        
        return np.stack(mask_list, axis=0), np.stack(mask_hex_list, axis=0)
        # self.retina.raw_image_to_hex_pxls(filtered_raw)

    def get_obstacle_pos(self, hex_pxls, raw_vision, updated) :
        
        if updated :
            left_rgb, right_rgb = self.get_human_readable(hex_pxls)

            left = left_rgb.max(axis=2)
            right = right_rgb.max(axis=2)

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
            
            monitor_zone_has_obstacle = (monitor_zone > 5).any()
            
            # get the mean of second axis of where obstacle
            # obstacle_positions = np.where(np.min(self.history_detection_zone, axis=0) < 250)
            obstacle_positions = np.where(detection_zone > 5)
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
            self.visualization(left_rgb, right_rgb, raw_vision, self.last_obstacle_position, self.doging)
            return result
        else :
            assert self.last_result is not None, "No previous result available"
            return self.last_result

class MyDataSet(torch.utils.data.Dataset) :
    
    def __init__(self, data_dir, transform=None, stack_size=3) :
        self.data_dir = data_dir
        self.transform = transform
        self.stack_size = stack_size
        self.data = []
        self.possible_idx = []
        
        for dir in os.listdir(data_dir) :
            if os.path.isdir(dir) :
                tmp_data = []
                for file in os.listdir(dir) :
                    if file.endswith(".npz") :
                        tmp_data.append(os.path.join(dir, file))
                self.possible_idx += [len(self.data) + i for i in range(len(tmp_data) - stack_size + 1)]
                self.data += tmp_data
    
    def __len__(self) :
        return len(self.data)
    
    def __getitem__(self, idx) :
        data = np.load(self.data[idx])
        import ipdb

        ipdb.set_trace()
        if self.transform :
            data = self.transform(data)
        return data


if __name__ == "__main__" :

    # Train the vision encoder
    dataset = MyDataSet(
        data_dir="outputs/data",
        transform=None,
        stack_size=3
    )

    print(dataset[0])
