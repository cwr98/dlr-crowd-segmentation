import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_pipeline.dataset import CrowdDataset
from src.models.model import SimpleSegNet


def main():
    # choose CPU or GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # load dataset
    dataset = CrowdDataset()

    # load batches
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # create model
    model = SimpleSegNet()
    model = model.to(device)

    # loss function for binary segmentation
    loss_fn = nn.BCEWithLogitsLoss()

    # optimizer updates model weights
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(epochs):
        total_loss = 0.0

        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            # forward pass
            outputs = model(images)

            # compare prediction to true mask
            loss = loss_fn(outputs, masks)

            # clear old gradients
            optimizer.zero_grad()

            # compute gradients
            loss.backward()

            # update weights
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # save trained model
    torch.save(model.state_dict(), "models/simple_segnet.pth")
    print("Saved model to models/simple_segnet.pth")


if __name__ == "__main__":
    main()