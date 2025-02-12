import time
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision.transforms import transforms, autoaugment
from sentence_transformers import SentenceTransformer

from abstract_trainer import ClrTrainer
from util import caption_from_labels

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class XClrTrainer(ClrTrainer):
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
        augmentation = transforms.Compose(
            [
                autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),
                transforms.Lambda(lambd=lambda x: x / 255.0),
            ]
        )

        initial_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.PILToTensor(),
            ]
        )

        super().__init__(
            dataset_path=dataset_path ,
            batch_size=batch_size,
            device=device,
            encoder_checkpoint_path=encoder_checkpoint_path,
            head_out_features=head_out_features,
            num_worker_dl=num_worker_dl,
            epochs=epochs,
            encoder_load_path=encoder_load_path,
            initial_transform=initial_transform,
            image_augmentation_transform=augmentation,
        )

        self._tau = tau
        self._tau_s = tau_s
        self._labels = [i for i in range(label_range)]
        self._compute_similarity_graph()


    def _compute_similarity_graph(self):
        caption_encoder = SentenceTransformer("all-MiniLM-L6-v2").eval()
        with torch.no_grad():
            captioned_labels = caption_from_labels(self._labels)
            encoded_captions = caption_encoder.encode(captioned_labels)
            self._similarity_graph = caption_encoder.similarity(encoded_captions, encoded_captions)
            self._similarity_graph = self._similarity_graph.to(self._device)


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
                labels = labels.repeat(2).to(self._device)
                augmented_images = self._double_aug(images)

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
        num_worker_dl=1,
    )
    trainer.train()
