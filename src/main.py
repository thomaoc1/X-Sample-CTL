import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.transforms import transforms, autoaugment
from sentence_transformers import SentenceTransformer


def init_data_loader(path: str, batch_size: int = 64) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.PILToTensor(),
    ])

    dataset = ImageFolder(
        root=path,
        transform=transform,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )


def init_models(out_features: int) -> tuple[SentenceTransformer, nn.Module, nn.Module]:
    head = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, out_features),
    )

    return (
        SentenceTransformer("all-MiniLM-L6-v2").eval(),
        resnet50(),
        head,
    )


def train(batch_size=64):
    augmentation = transforms.Compose([
        autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),
        transforms.Lambda(lambd=lambda x: x / 255.0),
    ])

    loader = init_data_loader('datasets/ImageNet-S-50/train', batch_size=batch_size)
    caption_encoder, image_encoder, head = init_models(out_features=2 * batch_size)

    optimiser = optim.Adam(
        list(image_encoder.parameters()) + list(head.parameters()),
        lr=1e-4,
    )

    epochs = 10
    for epoch in range(epochs):
        for images, labels in loader:
            # (1) Augment images twice and concat
            # (2) Create similarity graph G using labels (captions)
            #       => Repeat labels (dog, cat, ... mouse, dog, cat, ..., mouse)
            #       => Since captions are invariant, maybe construct graph before training (?)
            # (3) Apply column-wise softmax to G
            # (4) Forward pass stacked images
            # (5) Compute loss
            #       => Dimension of head output is 2Nx2N where each column is the predicted similarity p
            #       => Cross-Entropy loss between G and p
            # (6) Backprop
            pass


if __name__ == '__main__':
    train()