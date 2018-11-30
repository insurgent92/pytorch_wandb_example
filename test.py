import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def test(args, model, device, test_loader, wandb):
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Save the first inbput tensor in each test batch as an example image
            example_images.append(wandb.Image(data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Log the images and metrics
    wandb.log({
            "Examples": example_images,
            "Test Accuracy": 100. * correct / len(test_loader.dataset),
            "Test Loss": test_loss})
