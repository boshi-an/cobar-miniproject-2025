import torch
import os
from torch import nn
import torch.nn.functional as F
import numpy as np

class MyDataSet(torch.utils.data.Dataset) :
    
    def __init__(self, data_dir, transform=None, stack_size=3) :
        self.data_dir = data_dir
        self.transform = transform
        self.stack_size = stack_size
        self.data = []
        self.possible_idx = []
        
        for dir in os.listdir(data_dir) :
            sub_dir = os.path.join(data_dir, dir)
            if os.path.isdir(sub_dir) :
                tmp_data = []
                files = os.listdir(sub_dir)
                # Sort files by number in the name
                files.sort(key=lambda x: int(x.split(".")[0]))
                for file in files :
                    if file.endswith(".npz") :
                        tmp_data.append(os.path.join(sub_dir, file))
                self.possible_idx += [len(self.data) + i for i in range(len(tmp_data) - stack_size + 1)]
                self.data += tmp_data
    
    def __len__(self) :
        return len(self.possible_idx)
    
    def __getitem__(self, idx) :
        stacked_data = {}
        for i in range(self.stack_size) :
            frame_idx = i + self.possible_idx[idx]
            npz_file = np.load(self.data[frame_idx])
            data = {key: npz_file[key] for key in npz_file.files}
            if self.transform :
                data = self.transform(data)
            for key in data :
                if key not in stacked_data :
                    stacked_data[key] = []
                stacked_data[key].append(data[key])
        for key in stacked_data :
            stacked_data[key] = torch.from_numpy(np.stack(stacked_data[key], axis=0))
        return stacked_data

class DoubleConv(nn.Module):
    """Two 3x3 convolutions with ReLU and optional BatchNorm"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class CNN(nn.Module) :
    
    def __init__(self, input_channels=1, stack_frames=3, h=256, w=450, features=[16, 32, 64, 128, 256, 512], device="cpu") :

        super().__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.device = device

        # Input channels
        in_channels = input_channels * stack_frames
        out_channels = in_channels

        # Downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Loss
        self.loss_func1 = nn.MSELoss()
        self.loss_func2 = nn.BCELoss()

        # self._initialize_weights()
    
    def _preprocess_human(self, vision_human) :

        vision_human_combined = torch.cat([vision_human[:, :, 0], vision_human[:, :, 1]], dim=3)
        # vision_human_combined:
        # 0: batch
        # 1: history
        # 2: height
        # 3: width
        # 4: channel
        vision_human_combined_stacked = torch.permute(vision_human_combined, (0, 2, 3, 1, 4))
        vision_human_combined_stacked = vision_human_combined_stacked.max(dim=4).values
        vision_human_combined_stacked = torch.permute(vision_human_combined_stacked, (0, 3, 1, 2)).contiguous()
        return vision_human_combined_stacked
    
    def _dice_loss(self) :

        def f(pred, target) :
            intersection = (pred * target).sum(axis=(1, 2, 3))
            loss = 1 - (2 * intersection + 1e-9) / (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3)) + 1e-9)

            return loss.mean()
        
        return f

    def _forward_pass(self, x) :

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor, dict]:

        vision_human = data["human_readable"].to(self.device).float() / 255
        mask_human = data["mask"].to(self.device).float() / 255

        vision_human_processed = self._preprocess_human(vision_human)
        mask_human_processed = self._preprocess_human(mask_human)
        mask_human_processed = torch.where(
            mask_human_processed > 0.5,
            torch.ones_like(mask_human_processed, device=mask_human_processed.device),
            torch.zeros_like(mask_human_processed, device=mask_human_processed.device)
        )

        out = self._forward_pass(vision_human_processed)
        out_sigmoid = torch.sigmoid(out)

        loss1 = self.loss_func1(out, mask_human_processed)
        loss2 = 0 #self.loss_func2(out_sigmoid, mask_human_processed)
        loss = loss1
        return out, loss, {"loss1": loss1, "loss2": loss2}

    def predict_single_frame(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        vision_human = data["human_readable"].to(self.device).float() / 255
        vision_human = vision_human.unsqueeze(0)
        vision_human_processed = self._preprocess_human(vision_human)

        if "mask" in data :
            mask_human = data["mask"].to(self.device).float() / 255
            mask_human = mask_human.unsqueeze(0)
            mask_human_processed = self._preprocess_human(mask_human).squeeze(0)
            mask_human_return = mask_human_processed
        else :
            mask_human_return = None

        out = self._forward_pass(vision_human_processed)
        out = out.squeeze(0)
        vision_human_processed = vision_human_processed.squeeze(0)

        return out, vision_human_processed, mask_human_return

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
