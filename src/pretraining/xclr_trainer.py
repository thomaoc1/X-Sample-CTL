import torch
import os
import torch.nn as nn
from torchvision.transforms import transforms, autoaugment
from sentence_transformers import SentenceTransformer

from src.pretraining.abstract_trainer import ClrTrainer
from src.util import caption_from_labels

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class XClrTrainer(ClrTrainer):
    def __init__(
            self,
            dataset_path: str,
            batch_size: int,
            device: str,
            label_range: int,
            encoder_checkpoint_base_path: str,
            head_out_features: int = 128,
            tau: float = 0.1,
            tau_s: float = 0.1,
            num_workers_dl: int = 8,
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
            encoder_checkpoint_base_path=encoder_checkpoint_base_path,
            tau=tau,
            head_out_features=head_out_features,
            num_workers_dl=num_workers_dl,
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

    def _compute_loss(self, **kwargs):
        labels, encoding_similarities = kwargs['labels'], kwargs['encoding_similarities']
        labels = labels.repeat(2).to(self._device)
        sub_sim_graph = self._similarity_graph[labels][:, labels]
        target = nn.functional.softmax(sub_sim_graph / self._tau_s, dim=1)
        return nn.functional.cross_entropy(encoding_similarities / self._tau, target)


if __name__ == '__main__':
    trainer = XClrTrainer(
        batch_size=256,
        label_range=50,
        dataset_path='datasets/ImageNet-S-50/train',
        encoder_checkpoint_base_path='checkpoints/b256-AdamW-3e-4-CosineAnnealing',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_workers_dl=8,
    )
    trainer.train()
