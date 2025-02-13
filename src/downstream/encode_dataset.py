import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.fx.node import base_types
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.pretraining.encoder import ResNetEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('encoder_weights_path', type=str, help='Path to weights for encoder')
    parser.add_argument('model', choices=['xclr', 'simclr'], type=str, help='Model used for encoder training')
    parser.add_argument('model_id', type=str, help='Unique identifier for trained model (ex. b256_AdamW_3e-4)')
    return parser.parse_args()


def init_cifar_loaders():
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, test_loader


def extract_features_dataset(dataloader: DataLoader, encoder: nn.Module, device: str):
    encodings_list, labels_list = [], []
    for img, label in tqdm(dataloader, total=len(dataloader), desc="Extracting Features"):
        with torch.no_grad():
            encodings = encoder(img.to(device)).flatten(1)
        encodings_list.append(encodings.cpu())
        labels_list.append(label.cpu())
    encodings = torch.cat(encodings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return encodings, labels


def init_encoder(path: str, device: str):
    image_encoder = ResNetEncoder(detach_head=True).to(device)
    image_encoder.load_state_dict(
        torch.load(
            path,
            weights_only=True,
            map_location=device
        )
    )

    image_encoder.eval()
    image_encoder.requires_grad_(False)
    return image_encoder


def save_encoding_label_pairs(encodings, labels, base_save_path: str, filename: str):
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    torch.save(
        {
            'encodings': encodings,
            'labels': labels,
        },
        os.path.join(base_save_path, filename)
    )


def encode_dataset(encoder_weight_path: str, model: str, model_id: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_encoder = init_encoder(path=encoder_weight_path, device=device)
    train_loader, test_loader = init_cifar_loaders()
    train_encodings, train_labels = extract_features_dataset(dataloader=train_loader, encoder=image_encoder, device=device)
    test_encodings, test_labels = extract_features_dataset(dataloader=test_loader, encoder=image_encoder, device=device)

    base_save_path = os.path.join('datasets/encoded/', model, model_id)
    save_encoding_label_pairs(train_encodings, train_labels, base_save_path, 'train.pt')
    save_encoding_label_pairs(test_encodings, test_labels, base_save_path, 'test.pt')


if __name__ == '__main__':
    args = parse_args()

    encode_dataset(
        encoder_weight_path=args.encoder_weights_path,
        base_save_path=args.base_save_path,
        model_id=args.model_id
    )
