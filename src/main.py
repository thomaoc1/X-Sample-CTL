import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchlars import LARS
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.transforms import transforms, autoaugment
from sentence_transformers import SentenceTransformer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        num_workers=4,
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
        SentenceTransformer("all-MiniLM-L6-v2").eval(),
        nn.Sequential(*list(encoder.children())[:-1]).to(device),
        head.to(device),
    )


def compute_similarity_graph(labels: list, encoder: SentenceTransformer):
     with torch.no_grad():
         captioned_labels = caption_from_labels(labels)
         encoded_captions = encoder.encode(captioned_labels)
         return encoder.similarity(encoded_captions, encoded_captions)


def train(class_labels: list, checkpoint_path: str, batch_size=1024, tau=0.1, device='cpu'):
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

    #base_optimizer = optim.SGD(list(image_encoder.parameters()) + list(head.parameters()), lr=7.5e-2)
    #optimiser = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.005)
    optimiser = optim.AdamW(list(image_encoder.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=1e-4)

    scaler = GradScaler()

    epochs = 30 
    epoch_losses = []
    print('Training', flush=True)
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in loader:
            optimiser.zero_grad()
            # (1) Augment images twice and concat
            images = images.to(device)
            augmented_images = torch.concat(
                [ augmentation(images), augmentation(images) ],
                dim=0,
            )

            labels = labels.repeat(2).to(device)

            # (2) Extract sim graph for labels
            sub_sim_graph = similarity_graph[labels][:, labels]

            # (3) Apply column-wise softmax to G
            softmax_sim_graph = nn.functional.softmax(sub_sim_graph / tau, dim=1)

            # (4) Forward pass
            with autocast(dtype=torch.float16):
                backbone_encoding = image_encoder(augmented_images).flatten(1)
                image_encodings = head(backbone_encoding)

                image_encodings = nn.functional.normalize(image_encodings, p=2, dim=1)
                image_sim_graph = image_encodings @ image_encodings.T

            # (5) Compute loss
            loss = nn.functional.cross_entropy(image_sim_graph / tau, softmax_sim_graph)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}", flush=True)

        torch.save(
            image_encoder.state_dict(),
            checkpoint_path + '-resnet_encoder.pt',
        )

        torch.save(
            head.state_dict(),
            checkpoint_path + '-head_parameters.pt',
        )


if __name__ == '__main__':
    train(
        class_labels=[i for i in range(50)],
        checkpoint_path='checkpoints/b256-AdamW-3e-4',
        batch_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )  
