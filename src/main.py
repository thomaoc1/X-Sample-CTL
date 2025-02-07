import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.transforms import transforms, autoaugment
from sentence_transformers import SentenceTransformer

from util import caption_from_labels


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


def init_models(out_features: int, device: str) -> tuple[SentenceTransformer, nn.Module, nn.Module]:
    head = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, out_features),
    )

    encoder = resnet50()

    return (
        SentenceTransformer("all-MiniLM-L6-v2").to(device),
        nn.Sequential(*list(encoder.children())[:-1]).to(device),
        head.to(device),
    )


def compute_similarity_graph(labels: list, encoder: SentenceTransformer):
     with torch.no_grad():
         captioned_labels = caption_from_labels(labels)
         encoded_captions = encoder.encode(captioned_labels)
         return encoder.similarity(encoded_captions, encoded_captions)


def train(class_labels: list, batch_size=1024, tau=1, device='cpu'):
    augmentation = transforms.Compose([
        autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),
        transforms.Lambda(lambd=lambda x: x / 255.0),
    ])

    loader = init_data_loader('datasets/ImageNet-S-50/train', batch_size=batch_size)
    caption_encoder, image_encoder, head = init_models(
        out_features=128, device=device
    )

    similarity_graph = compute_similarity_graph(class_labels, caption_encoder) / tau
    similarity_graph = similarity_graph.to(device)

    optimiser = optim.Adam(
        list(image_encoder.parameters()) + list(head.parameters()),
        lr=7.5e-2,
    )

    epochs = 100
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in loader:
            # (1) Augment images twice and concat
            augmented_images = torch.concat(
                [ augmentation(images), augmentation(images) ],
                dim=0,
            ).to(device)

            labels = labels.repeat(2).to(device)

            # (2) Extract sim graph for labels
            sub_sim_graph = similarity_graph[labels][:, labels]

            # (3) Apply column-wise softmax to G
            softmax_sim_graph = nn.functional.softmax(sub_sim_graph, dim=1)

            # (4) Forward pass
            backbone_encoding = image_encoder(augmented_images).flatten(1)
            image_encodings = head(backbone_encoding)

            image_encodings = nn.functional.normalize(image_encodings, p=2, dim=1)
            image_sim_graph = image_encodings @ image_encodings.T

            # (5) Compute loss
            loss = nn.functional.cross_entropy(image_sim_graph / tau, softmax_sim_graph)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        torch.save(
            image_encoder.parameters(),
            'resnet_encoder.pt',
        )

        torch.save(
            head.parameters(),
            'head_parameters.pt',
        )

    print(losses, flush=True)


if __name__ == '__main__':
    train(
        class_labels=[i for i in range(50)],
        batch_size=4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )