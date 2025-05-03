import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from submission_level0.cnn import MyDataSet, CNN
import argparse
import os

if __name__ == "__main__" :

    # Define arguments
    parser = argparse.ArgumentParser(description="Train a vision encoder model.")
    parser.add_argument("--data_path", type=str, default="outputs/data", help="Path to the dataset.")
    parser.add_argument("--save_path", type=str, default="outputs/cnn", help="Path to save the trained model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--stack_frames", type=int, default=3, help="Stack how many frames.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Stack how many frames.")
    args = parser.parse_args()

    # Train the vision encoder
    dataset = MyDataSet(
        data_dir=args.data_path,
        transform=None,
        stack_size=3
    )

    # Split train and test dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    device = None
    if torch.cuda.is_available() :
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() :
        device = torch.device("mps")
    else :
        device = torch.device("cpu")

    model = CNN(stack_frames=3, device=device)
    model.to(device)
    print(f"Using device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    for epoch in range(args.epochs) :
        with tqdm(train_dataloader, desc=f"Epoch {epoch}", unit="batch") as pbar:
            for i, data in enumerate(pbar):
                optimizer.zero_grad()
                output, loss, info = model(data)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"Loss": info["loss1"].item()})
        tot_test_loss = 0
        num_test_batches = len(test_dataloader)
        with tqdm(test_dataloader, desc=f"Test {epoch}", unit="batch") as pbar:
            for i, data in enumerate(pbar):
                with torch.no_grad():
                    output, loss, info = model(data)
                    tot_test_loss += info["loss1"].item()
        avg_test_loss = tot_test_loss / num_test_batches
        print(f"Average test loss: {avg_test_loss:.4f}")
        # Save the model
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        torch.save(model.state_dict(), os.path.join(args.save_path, f"model_epoch_{epoch+1}.pth"))
        print(f"Model saved at epoch {epoch+1}")
    print("Training complete.")
