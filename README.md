# X-CLR, SimCLR Implementations
This work implements X-CLR [1] and SimCLR [2] using PyTorch. We used ImageNet-S-50 [3] when testing with various downstream tasks but the code is modular enough to allow for any dataset which implements the `src.pretraining.dataset_types.ValidClrDataset` interface.


## Python Environment
```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

## Organizing the Dataset

Create a `datasets/` directory at the root level and move `ImageNet-S-50` into it. The final structure should look like this:

<pre>
datasets/
└── ImageNet-S-50/ 
    └── train/
</pre>

To verify, run:

```bash
python test_dataset_setup.py
```

## Saving Checkpoints

Create a `checkpoints/` directory at the root level. 

## Training

### Prerequisites
It is extremely simple to add any valid dataset by ensuring it implements the 
interface `src.pretraining.dataset_types.valid_clr_dataset.ValidClrDatset`. Example implementations can be found in `src/pretraining/dataset_types`. 

For now, only `src.pretraining.dataset_types.image_net_s.ImageNetS` can be used by calling (for example):

```bash
python -m python -m src.pretraining.train xclr imagenet-s datasets/ImageNet-S-50/train -b 256 -nw 8
```

Full usage can be consulted through:
```bash
python -m src.pretraining.train --help
```

Once training completes, you will find your encoder checkpoint in `checkpoints/encoder/{alg}/{time}/encoder.pt` along with saved epoch losses in csv format.

## Downstream Tasks

### Classification 
For now, only the CIFAR-10 dataset is used for testing classification. 

First step is to encode the dataset using trained encoder weights as follows:

```bash
python -m src.downstream.encode_dataset.py path/to/model/checkpoint model_name model_id
```

Where model_id is a unique identifier like `b256_AdamW_3e-4`. The encoded dataset will be found in `datasets/encoded/{model}/{model_id}`

Next, for classification on the encoded dataset:
```bash
python -m src.downstream.classification.classifier path/to/encoded/dataset [--save]
```

This will output an accuracy score and save you classifier in `checkpoints/classifiers/path.to.encoded.dataset/`

## References

[1] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).  
*A simple framework for contrastive learning of visual representations.*  
In *International Conference on Machine Learning* (pp. 1597–1607). PMLR.  

[2] Sobal, V., Ibrahim, M., Balestriero, R., Cabannes, V., Bouchacourt, D., Astolfi, P., Cho, K., & LeCun, Y. (2024).  
*X-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs.*  
*arXiv preprint arXiv:2407.18134*.  


