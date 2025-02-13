import torch
from torchvision import transforms

from src.pretraining.abstract_trainer import ClrTrainer


class SimClrTrainer(ClrTrainer):

    @staticmethod
    def image_augmentation_fn(size: int):
        kernel_size = max(1, int(0.1 * size))  # Ensure it's at least 1
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # Make it odd
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=kernel_size),
                transforms.Lambda(lambd=lambda x: x / 255.0)
            ]
        )

    def __init__(
            self,
            dataset_path: str,
            batch_size: int,
            device: str,
            encoder_checkpoint_base_path: str,
            tau: float = 0.07,
            head_out_features: int = 128,
            num_workers_dl: int = 4,
            epochs: int = 100,
            encoder_load_path: str | None = None,
            resize: int = 224,
    ):
        initial_transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.RandomResizedCrop(resize, antialias=True),
            ]
        )

        super().__init__(
            dataset_path=dataset_path,
            batch_size=batch_size,
            device=device,
            encoder_checkpoint_base_path=encoder_checkpoint_base_path,
            tau=tau,
            head_out_features=head_out_features,
            num_workers_dl=num_workers_dl,
            epochs=epochs,
            encoder_load_path=encoder_load_path,
            initial_transform=initial_transform,
            image_augmentation_transform=SimClrTrainer.image_augmentation_fn(resize),
        )

    def _compute_loss(self, **kwargs) -> torch.Tensor:
        encoding_similarities = kwargs['encoding_similarities']

        # In case last batch < batch size
        current_batch_size = int(encoding_similarities.size(0) / 2)

        labels = torch.arange(current_batch_size).repeat(2).to(self._device)
        labels = labels.unsqueeze(dim=0) == labels.unsqueeze(dim=1)
        mask = torch.eye(labels.size(0), dtype=torch.bool)
        labels = labels[~mask].reshape(labels.size(0), -1).float()
        encoding_similarities = encoding_similarities[~mask].view(encoding_similarities.size(0), -1)

        positives = encoding_similarities[labels.bool()].view(labels.size(0), -1)
        negatives = encoding_similarities[~labels.bool()].view(encoding_similarities.size(0), -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(self._device)

        return torch.nn.functional.cross_entropy(logits / self._tau, labels)

if __name__ == '__main__':
    trainer = SimClrTrainer(
        dataset_path='datasets/ImageNet-S-50/train',
        batch_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        encoder_checkpoint_base_path='checkpoints/encoders/b256-simclr.pt',
        num_workers_dl=8,
    )

    trainer.train()


