import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from tqdm import tqdm
from submission_level0.cnn import MyDataSet, CNN
import argparse
import numpy as np
import cv2

def get_merged_gt(data, gt) :

    mergedl = np.concatenate((data["human_readable"][0,0], data["human_readable"][1,0], data["human_readable"][2,0]), axis=0)
    mergedr = np.concatenate((data["human_readable"][0,1], data["human_readable"][1,1], data["human_readable"][2,1]), axis=0)
    merged = np.concatenate((mergedl, mergedr), axis=1) / 255
    merged2 = np.concatenate((gt[0], gt[1], gt[2]), axis=0)
    merged_all = merged.copy()
    merged_all[:, :, 2] = merged2

    return merged_all

if __name__ == "__main__" :

    # Define arguments
    parser = argparse.ArgumentParser(description="Train a vision encoder model.")
    parser.add_argument("--data_path", type=str, default="outputs/test_data", help="Path to the dataset.")
    parser.add_argument("--load_path", type=str, default="outputs/cnn/model_epoch_5.pth", help="Path to load the trained model.")
    args = parser.parse_args()

    # Train the vision encoder
    dataset = MyDataSet(
        data_dir=args.data_path,
        transform=None,
        stack_size=3
    )

    model = CNN(device="cpu")
    model.load_state_dict(torch.load(args.load_path))
    model.to("cpu")
    model.eval()
    with torch.no_grad() :
        for i in range(len(dataset)) :
            data = dataset[i]

            out, base, gt = model.predict_single_frame(data)

            merged_gt = get_merged_gt(data, gt.cpu().numpy())
            merged_out = get_merged_gt(data, out.cpu().numpy())
            merged_all = np.concatenate((merged_gt, merged_out), axis=1)

            cv2.imshow("Merged", merged_all)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
