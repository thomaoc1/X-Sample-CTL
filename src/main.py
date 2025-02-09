import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import GradScaler, autocast
# from torchlars import LARS
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, autoaugment

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from image_encoder import ImageEncoder
from util import caption_from_labels


class XClrTrainer:
    def __init__(
            self,
            dataset_path: str,
            num_classes: int,
            batch_size: int,
            tau: float = 0.1,
            tau_s: float = 0.1,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self._device = device
        self._tau = tau
        self._tau_s = tau_s
        self._class_labels = [i for i in range(num_classes)]
        self._augmentation = transforms.Compose(
            [
                autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),
                transforms.Lambda(lambd=lambda x: x / 255.0),
            ]
        )
        self._compute_similarity_graph()
        self._init_data_loader(path=dataset_path, batch_size=batch_size)
        self._init_model()
        self._init_optimiser()

    def _init_data_loader(self, path: str, batch_size):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.PILToTensor(),
            ]
        )

        dataset = ImageFolder(
            root=path,
            transform=transform,
        )

        self._loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

    def _init_model(self):
        self._image_encoder = nn.DataParallel(
            ImageEncoder(out_features=128)
        ).to(self._device)

    def _init_optimiser(self):
        # base_optimizer = optim.SGD(list(image_encoder.parameters()) + list(head.parameters()), lr=7.5e-2)
        # optimiser = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.005)
        self._optimiser = optim.AdamW(
            self._image_encoder.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
        )

    def _compute_similarity_graph(self):
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        with torch.no_grad():
            captioned_labels = caption_from_labels(self._class_labels)
            encoded_captions = encoder.encode(captioned_labels)
            self._similarity_graph = encoder.similarity(encoded_captions, encoded_captions) / self._tau_s

    def _double_aug(self, images, labels):
        labels = labels.repeat(2).to(self._device)
        images = images.to(self._device)
        augmented_images = torch.concat(
            [self._augmentation(images), self._augmentation(images)],
            dim=0,
        )
        return augmented_images, labels

    def _save_model_state(self):
        torch.save(
            self._image_encoder.state_dict(),
            'image_encoder.pt',
        )

    def train(self, epochs: int = 100):
        scaler = GradScaler()
        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for images, labels in self._loader:
                self._optimiser.zero_grad()
                # (1) Augment images twice and concat
                augmented_images, labels = self._double_aug(images, labels)
                # (2) Extract sim graph for labels
                sub_sim_graph = self._similarity_graph[labels][:, labels]
                # (3) Apply column-wise softmax to G
                softmax_sim_graph = nn.functional.softmax(sub_sim_graph / self._tau_s, dim=1)
                # (4) Forward pass
                with autocast():
                    image_encodings = self._image_encoder(augmented_images)
                    image_encodings = nn.functional.normalize(image_encodings, p=2, dim=1)
                    image_sim_graph = image_encodings @ image_encodings.T
                # (5) Compute loss (CE requires 32 bit precision for stability)
                loss = nn.functional.cross_entropy(image_sim_graph / self._tau, softmax_sim_graph)
                # (6) Scale and BP
                scaler.scale(loss).backward()
                scaler.step(self._optimiser)
                scaler.update()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self._loader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}", flush=True)

            self._save_model_state()


if __name__ == '__main__':
    print(f"Available GPUs: {torch.cuda.device_count()}", flush=True)

    trainer = XClrTrainer(
        dataset_path='datasets/ImageNet-S-50',
        num_classes=50,
        batch_size=1024,
    )
    trainer.train()
