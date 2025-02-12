import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50
from tqdm import tqdm

from encoder import ResNetEncoder


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


def extract_features_dataset(dataloader: DataLoader, encoder: nn.Module):
    encodings_list, labels_list = [], []
    for img, label in tqdm(dataloader, total=len(dataloader), desc="Extracting Features"):
        with torch.no_grad():
            encodings = encoder(img).flatten(1)
        encodings_list.append(encodings)
        labels_list.append(label)
    encodings = torch.cat(encodings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return encodings, labels

def load_simclr(device: str, path: str):
    checkpoint_dict = torch.load(
        path,
        weights_only=False,
        map_location=device,
    )
    model_weights = checkpoint_dict['state_dict']
    model = ResNetEncoder(out_dim=128)
    model.load_state_dict(model_weights)
    model._backbone.fc = nn.Identity()
    return model


def init_encoder(path: str, is_simclr=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if is_simclr:
        image_encoder = load_simclr(device, 'checkpoints/encoders/b256-simclr.pt')
    else:
        image_encoder = resnet50()
        image_encoder = nn.Sequential(*list(image_encoder.children())[:-1])

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

def save_encoding_label_pairs(encodings, labels, path: str):
    torch.save(
        {
            'encodings': encodings,
            'labels': labels,
        },
        path,
    )



def encode_dataset():
    image_encoder = init_encoder(path='checkpoints/encoders/b256-simclr.pt', is_simclr=True)
    train_loader, test_loader = init_cifar_loaders()
    train_encodings, train_labels = extract_features_dataset(dataloader=train_loader, encoder=image_encoder)
    test_encodings, test_labels = extract_features_dataset(dataloader=test_loader, encoder=image_encoder)
    base_path = 'datasets/encoded/simclr/encoded_cifar10_'
    save_encoding_label_pairs(train_encodings, train_labels, path=base_path + 'train.pt')
    save_encoding_label_pairs(test_encodings, test_labels, path=base_path + 'test.pt')


if __name__ == '__main__':
    encode_dataset()