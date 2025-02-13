import torch

from src.pretraining.abstract_trainer import ClrTrainer
from src.pretraining.dataset_types.dataset import ValidClrDataset


class SimClrTrainer(ClrTrainer):

    def __init__(
            self,
            dataset: ValidClrDataset,
            device: str,
            encoder_checkpoint_base_path: str,
            tau: float = 0.07,
            head_out_features: int = 128,
            epochs: int = 100,
            encoder_load_path: str | None = None,
    ):

        super().__init__(
            dataset=dataset,
            device=device,
            encoder_checkpoint_base_path=encoder_checkpoint_base_path,
            tau=tau,
            head_out_features=head_out_features,
            epochs=epochs,
            encoder_load_path=encoder_load_path,
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

