import os
import time
import pandas as pd
from abc import abstractmethod
from datetime import datetime
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast

from src.pretraining.dataset_types.valid_clr_dataset import ValidClrDataset
from src.pretraining.encoder import ResNetEncoder


class ClrTrainer:
    def __init__(
            self,
            dataset: ValidClrDataset,
            device: str,
            encoder_checkpoint_base_path: str,
            tau: float,
            head_out_features: int = 128,
            epochs: int = 100,
            encoder_load_path: str | None = None,
    ):
        self._device = device
        self._epochs = epochs
        self._tau = tau

        self._data_loader = dataset.get_dataloader()
        self._image_augmentation_fn = dataset.get_gpu_augmentations()

        self._init_checkpoint_dir(base_path=encoder_checkpoint_base_path)
        self._init_encoder(out_features=head_out_features, load_path=encoder_load_path)

    def _init_checkpoint_dir(self, base_path: str):
        now = datetime.now()
        dir_name = now.strftime("%b%d-%H:%M:%S")
        self._encoder_checkpoint_path = os.path.join(base_path, dir_name)
        if not os.path.exists(self._encoder_checkpoint_path):
            os.makedirs(self._encoder_checkpoint_path)

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

    def _double_aug(self, images: torch.Tensor) -> torch.Tensor:
        augmented_images = torch.concat(
            [self._image_augmentation_fn(images), self._image_augmentation_fn(images)],
            dim=0,
        )
        return augmented_images

    def _log(self, epoch: int, epoch_loss: float, start_time: float):
        avg_loss = epoch_loss / len(self._data_loader)
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(
            f"Epoch {epoch + 1}/{self._epochs} - Loss: {avg_loss:.4f} - Time Taken {minutes:02}:{seconds:02}min",
            flush=True
        )

    def _save_state(self, optimiser: torch.optim.Optimizer):
        torch.save(
            self._image_encoder.state_dict(),
            os.path.join(self._encoder_checkpoint_path, 'encoder.pt')
        )

        torch.save(
            optimiser.state_dict(),
            os.path.join(self._encoder_checkpoint_path, 'optimiser.pt'),
        )

    def _save_losses(self, losses: list):
        df = pd.DataFrame({"Loss": losses})
        df.to_csv(os.path.join(self._encoder_checkpoint_path, "losses.csv"), index=False)

    @abstractmethod
    def _compute_loss(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by child class")

    def train(self):
        optimiser = optim.AdamW(self._image_encoder.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=len(self._data_loader))

        print('=== Starting Training ===', flush=True)
        scaler = GradScaler()
        epoch_losses = []
        for epoch in range(self._epochs):
            epoch_loss = 0
            start = time.time()
            for images, labels in self._data_loader:
                optimiser.zero_grad()

                images = images.to(self._device)
                augmented_images = self._double_aug(images)

                with autocast(dtype=torch.float16):
                    image_encodings = self._image_encoder(augmented_images)
                    image_encodings = nn.functional.normalize(image_encodings, p=2, dim=1)
                    encoding_similarities = image_encodings @ image_encodings.T

                loss = self._compute_loss(encoding_similarities=encoding_similarities, labels=labels)

                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()

                epoch_loss += loss.item()

            if epoch >= 15:
                scheduler.step()

            epoch_losses.append(epoch_loss)
            self._log(epoch=epoch, epoch_loss=epoch_loss, start_time=start)
            self._save_state(optimiser=optimiser)

        self._save_losses(losses=epoch_losses)

