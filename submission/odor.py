import numpy as np
from collections import deque

class OdorNavigator :
    
    def __init__(self, base_length=0.5, history_length=10) :
        
        self.base_length = base_length
        
        self.history_odor_values = deque(maxlen=history_length)
        self.history_forward_speed = deque(maxlen=history_length)
    
    def get_odor_angle(self, odor_values) :
        
        self.history_odor_values.append(odor_values)
        smoothed_odor_values = np.mean(self.history_odor_values, axis=0)
        attractive_distance = np.average(
            1 / smoothed_odor_values[0].reshape(2, 2)**0.5, axis=0, weights=[9,1]
        )
        
        l = attractive_distance[0]
        r = attractive_distance[1]
        b = self.base_length
        
        ratio = np.clip((l-r) / b, -1, 1)
        alpha = np.arcsin(ratio)
        angle = np.clip(alpha, -np.pi/2, np.pi/2) / (np.pi/2)
        
        if self.is_behind() :
            if angle > 0 :
                angle = 2
            else :
                angle = -2
        
        return angle
    
    def update(self, move_direction) :
        
        forward_speed = move_direction[0] + move_direction[1]
        self.history_forward_speed.append(forward_speed)
        
    def is_behind(self) :
        
        if len(self.history_forward_speed) < 2 :
            return False
        
        forward_speed = np.mean(self.history_forward_speed, axis=0)
        odor_change = self.history_odor_values[-1].mean() - self.history_odor_values[0].mean()
        
        if forward_speed > 0 and odor_change < 0 :
            return True
        else :
            return False
