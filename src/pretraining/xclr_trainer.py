import torch
import os
import torch.nn as nn
from torchvision.transforms import transforms, autoaugment
from sentence_transformers import SentenceTransformer

from src.pretraining.abstract_trainer import ClrTrainer
from src.pretraining.dataset_types.valid_clr_dataset import ValidClrDataset
from src.util import caption_from_labels

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class XClrTrainer(ClrTrainer):
    def __init__(
            self,
            dataset: ValidClrDataset,
            device: str,
            label_range: int,
            encoder_checkpoint_base_path: str,
            head_out_features: int = 128,
            tau: float = 0.1,
            tau_s: float = 0.1,
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

