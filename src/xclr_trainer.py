import time
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, autoaugment
from sentence_transformers import SentenceTransformer

from encoder import ResNetEncoder
from util import caption_from_labels

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class XClrTrainer:
    def __init__(
            self,
            dataset_path: str,
            batch_size: int,
            device: str,
            label_range: int,
            encoder_checkpoint_path: str,
            head_out_features: int = 128,
            tau: float = 0.1,
            tau_s: float = 0.1,
            num_worker_dl: int = 8,
            epochs: int = 100,
            encoder_load_path: str | None = None,
    ):
        self._batch_size = batch_size
        self._epochs = epochs
        self._tau = tau
        self._tau_s = tau_s
        self._device = device
        self._labels = [i for i in range(label_range)]
        self._encoder_checkpoint_path = encoder_checkpoint_path
        self._init_data_loader(path=dataset_path, num_workers=num_worker_dl)
        self._init_encoder(out_features=head_out_features, load_path=encoder_load_path)
        self._compute_similarity_graph()

        self._augmentation = transforms.Compose(
            [
                autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),
                transforms.Lambda(lambd=lambda x: x / 255.0),
            ]
        )

    def _init_data_loader(self, path: str, num_workers: int):
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

        self._data_loader = DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )

    def _init_encoder(self, out_features: int, load_path=None):
        self._image_encoder = ResNetEncoder(out_dim=out_features).to(self._device)
        if load_path:
            try:
                self._image_encoder.load_state_dict(
                    torch.load(f"{load_path}-image_encoder.pt", map_location=self._device)
                )
                print("Loaded pre-trained weights successfully.")
            except FileNotFoundError:
                print("No pre-trained weights found. Training from scratch.")

    def _compute_similarity_graph(self):
        caption_encoder = SentenceTransformer("all-MiniLM-L6-v2").eval()
        with torch.no_grad():
            captioned_labels = caption_from_labels(self._labels)
            encoded_captions = caption_encoder.encode(captioned_labels)
            self._similarity_graph = caption_encoder.similarity(encoded_captions, encoded_captions)
            self._similarity_graph = self._similarity_graph.to(self._device)

    def _double_aug_and_labels(self, images: torch.Tensor, labels: torch.Tensor):
        augmented_images = torch.concat(
            [self._augmentation(images), self._augmentation(images)],
            dim=0,
        )
        return augmented_images, labels.repeat(2)

    def _log(self, epoch: int, epoch_loss: float, start_time: float):
        avg_loss = epoch_loss / len(self._data_loader)
        print(
            f"Epoch {epoch + 1}/{self._epochs} - Loss: {avg_loss:.4f} - Time Taken {((time.time() - start_time) / 60):2f}",
            flush=True
        )

    def _save(self, optimiser: torch.optim.Optimizer):
        torch.save(
            self._image_encoder.state_dict(),
            self._encoder_checkpoint_path + '-image_encoder.pt',
        )

        torch.save(
            optimiser.state_dict(),
            self._encoder_checkpoint_path + '-optimiser.pt',
        )

    def train(self):
        optimiser = optim.AdamW(self._image_encoder.parameters(), lr=3e-4, weight_decay=1e-4)

        print('=== Starting Training ===', flush=True)
        scaler = GradScaler()
        for epoch in range(self._epochs):
            epoch_loss = 0
            start = time.time()
            for images, labels in self._data_loader:
                optimiser.zero_grad()

                images = images.to(self._device)
                labels = labels.to(self._device)
                augmented_images, labels = self._double_aug_and_labels(images, labels)

                sub_sim_graph = self._similarity_graph[labels][:, labels]
                softmax_sim_graph = nn.functional.softmax(sub_sim_graph / self._tau_s, dim=1)

                with autocast(dtype=torch.float16):
                    image_encodings = self._image_encoder(augmented_images)
                    image_encodings = nn.functional.normalize(image_encodings, p=2, dim=1)
                    image_sim_graph = image_encodings @ image_encodings.T

                loss = nn.functional.cross_entropy(image_sim_graph / self._tau, softmax_sim_graph)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()

                epoch_loss += loss.item()

            self._log(epoch=epoch, epoch_loss=epoch_loss, start_time=start)
            self._save(optimiser=optimiser)


if __name__ == '__main__':
    trainer = XClrTrainer(
        batch_size=256,
        label_range=50,
        dataset_path='datasets/ImageNet-S-50/train',
        encoder_checkpoint_path='checkpoints/b256-AdamW-3e-4-CosineAnnealing',
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    trainer.train()
