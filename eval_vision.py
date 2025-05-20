import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from tqdm import tqdm
from submission.cnn import MyDataSet, CNN
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
    parser.add_argument("--load_path", type=str, default="outputs/cnn/model_epoch_20.pth", help="Path to load the trained model.")
    args = parser.parse_args()

    # Train the vision encoder
    dataset = MyDataSet(
        data_dir=args.data_path,
        transform=None,
        stack_size=3
    )

    print("Dataset size:", len(dataset))

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        model = CNN(device="cpu")
        model.load_state_dict(torch.load(args.load_path))
        model.to("cpu")
        model.eval()
        with torch.no_grad() :
            for i in range(len(dataset)) :
                data = dataset[i]

                out, base, gt = model.predict_single_frame(data)

                out = torch.where(out>0.5, out, 0)

                merged_gt = get_merged_gt(data, gt.cpu().numpy())
                merged_out = get_merged_gt(data, out.cpu().numpy())
                merged_all = np.concatenate((merged_gt, merged_out), axis=1)

                # Change dtype to uint8
                merged_all_int = (merged_all * 255).astype(np.uint8)

                cv2.imshow("Merged", merged_all_int)

                # Save the images as a video
                if i == 0 :
                    out_video = cv2.VideoWriter("outputs/videos/merged.mp4", fourcc, 30, (merged_all.shape[1], merged_all.shape[0]))
                    out_video.write(merged_all_int)
                else :
                    out_video.write(merged_all_int)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    finally :
        if 'out_video' in locals() and out_video.isOpened():
            out_video.release()
        cv2.destroyAllWindows()
        print("Video saved as outputs/test_data/merged.mp4")
