import torch
import os
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from flygym.vision import Retina
from eval_vision import get_merged_gt
from .cnn import MyDataSet, CNN
from collections import deque
from scipy.stats import linregress
import numpy as np
import cv2

class BallDetector :
    
    def __init__(self, ckpt_path, stack_frames=3, monitor_ball_change=10, device="cpu") :

        self.device = device
        self.stack_frames = stack_frames
        self.monitor_ball_change = monitor_ball_change
        self.retina = Retina()
        self.model = CNN(device=device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.last_result = None
        self.vision_history = deque(maxlen=stack_frames)
        self.visible_count = 0
        self.invisible_count = 0
        self.filtered_visible_count = 0
        self.ball_size = deque(maxlen=monitor_ball_change)

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

    def _get_stacked_history(self) :

        if len(self.vision_history) == 0 :
            return None
        padding = torch.zeros((self.stack_frames - len(self.vision_history), *self.vision_history[0].shape)).to(self.device)
        stacked_history = torch.stack(list(self.vision_history), dim=0) if len(self.vision_history) > 1 else self.vision_history[0].unsqueeze(0)
        stacked_history = torch.cat((padding, stacked_history), dim=0)
        return stacked_history
        
    def _update_visible(self, is_visible) :

        if is_visible :
            self.visible_count += 1
            self.invisible_count = 0
            if self.visible_count > 2 :
                self.filtered_visible_count += 1
        else :
            self.invisible_count += 1
            self.visible_count = 0
            if self.invisible_count > 2 :
                self.filtered_visible_count = 0

    def _check_order(self, dq):
        if len(dq) < 2:
            return "not enough data"

        y = list(dq)
        x = list(range(len(y)))
        
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        if slope > 0:
            return "ascending"
        elif slope < 0:
            return "descending"
        else:
            return "flat"

    def get_ball_pos(self, hex_pxls, raw_vision, updated) :

        if updated :

            human_readable = self.get_human_readable(hex_pxls)
            self.vision_history.append(torch.from_numpy(np.stack(human_readable, axis=0)).to(self.device).float())
            stacked_human_readable = self._get_stacked_history()
            data = {
                "human_readable": stacked_human_readable,
            }
            with torch.no_grad() :
                pred_mask, _, _ = self.model.predict_single_frame(data)
                pred_mask = pred_mask.cpu().numpy()

            is_ball = pred_mask[-1] > 0.5
            ball_y = np.nonzero(is_ball)[1]
            is_visible = is_ball.sum() > 5
            self._update_visible(is_visible)
            if is_visible :
                self.ball_size.append(ball_y.std())
                ball_change = self._check_order(self.ball_size)
                ball_y = (is_ball * np.arange(is_ball.shape[1]).reshape(1, -1)).mean()
                ball_y_scaled = ball_y * 2 / is_ball.shape[1] - 1
                if self.filtered_visible_count <= 14 and ball_change != "descending" :
                    if ball_y_scaled < 0 and ball_y_scaled > -0.5 :
                        result = (True, (1, -1))
                    elif ball_y_scaled > 0 and ball_y_scaled < 0.5 :
                        result = (True, (-1, 1))
                    else :
                        result = (True, (-1, -1))
                elif ball_change == "descending" :
                    # if ball is moving away
                    result = (False, (0, 0))
                elif self.filtered_visible_count > 14 :
                    # if already moved enough
                    result = (True, (0, 0))
                else :
                    raise ValueError("Invalid situation.")
                self.last_result = result
            else :
                self.ball_size.append(0)
                self.last_result = (False, (0, 0))
            
            return self.last_result

            # image = get_merged_gt(data, pred_mask)
            
        else :

            assert self.last_result is not None, "No previous result available"
            return self.last_result
