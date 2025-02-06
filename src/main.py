import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.transforms import transforms, autoaugment
from sentence_transformers import SentenceTransformer

from src.util import caption_from_labels


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


def init_models(in_features: int, out_features: int) -> tuple[SentenceTransformer, nn.Module, nn.Module]:
    head = nn.Sequential(
        nn.Linear(in_features, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, out_features),
    )

    encoder = resnet50()

    return (
        SentenceTransformer("all-MiniLM-L6-v2").eval(),
        nn.Sequential(*list(encoder.children())[:-1]),
        head,
    )


def train(batch_size=1024, tau=1):
    augmentation = transforms.Compose([
        autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),
        transforms.Lambda(lambd=lambda x: x / 255.0),
    ])

    loader = init_data_loader('datasets/ImageNet-S-50/train', batch_size=batch_size)
    caption_encoder, image_encoder, head = init_models(
        in_features=2048, out_features=2 * batch_size
    )

    optimiser = optim.Adam(
        list(image_encoder.parameters()) + list(head.parameters()),
        lr=1e-4,
    )

    epochs = 10
    for epoch in range(epochs):
        for images, labels in loader:
            # (1) Augment images twice and concat
            augmented_images = torch.concat(
                [ augmentation(images), augmentation(images) ],
                dim=0,
            )

            # (2) Create similarity graph G using labels (captions) divide by tau
            #       => Repeat labels (dog, cat, ... mouse, dog, cat, ..., mouse)
            #       => Since captions are invariant, maybe construct graph before training (?)
            repeated_labels = labels.repeat(2)
            captions = caption_from_labels(repeated_labels)

            encoded_captions = caption_encoder.encode(captions)
            sim_graph = caption_encoder.similarity(encoded_captions, encoded_captions)
            sim_graph = sim_graph / tau

            # (3) Apply column-wise softmax to G
            softmax_sim_graph = nn.functional.softmax(sim_graph, dim=1)


            # (4) Forward pass stacked images
            img_encoding = image_encoder(augmented_images).flatten(1)
            output = nn.functional.softmax(head(img_encoding), dim=1)


            # (5) Compute loss
            #       => Dimension of head output is 2Nx2N where each column is the predicted similarity p
            #       => Cross-Entropy loss between G and p
            loss = nn.functional.cross_entropy(output, softmax_sim_graph)

            # (6) Backprop
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        torch.save(
            image_encoder.parameters(),
            'resnet_encoder.pt',
        )

        torch.save(
            head.parameters(),
            'head_parameters.pt',
        )


if __name__ == '__main__':
    train()