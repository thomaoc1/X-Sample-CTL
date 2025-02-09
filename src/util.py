import torch
from sentence_transformers import SentenceTransformer

label_mapping = {
    0: 'goldfish',
    1: 'tiger shark',
    2: 'goldfinch',
    3: 'tree frog',
    4: 'kuvasz',
    5: 'red fox',
    6: 'siamese cat',
    7: 'american black bear',
    8: 'ladybug',
    9: 'sulphur butterfly',
    10: 'wood rabbit',
    11: 'hamster',
    12: 'wild boar',
    13: 'gibbon',
    14: 'african elephant',
    15: 'giant panda',
    16: 'airliner',
    17: 'ashcan',
    18: 'ballpoint',
    19: 'beach wagon',
    20: 'boathouse',
    21: 'bullet train',
    22: 'cellular telephone',
    23: 'chest',
    24: 'clog',
    25: 'container ship',
    26: 'digital watch',
    27: 'dining table',
    28: 'golf ball',
    29: 'grand piano',
    30: 'iron',
    31: 'lab coat',
    32: 'mixing bowl',
    33: 'motor scooter',
    34: 'padlock',
    35: 'park bench',
    36: 'purse',
    37: 'streetcar',
    38: 'table lamp',
    39: 'television',
    40: 'toilet seat',
    41: 'umbrella',
    42: 'vase',
    43: 'water bottle',
    44: 'water tower',
    45: 'yawl',
    46: 'street sign',
    47: 'lemon',
    48: 'carbonara',
    49: 'agaric'
}

def caption_from_label(label: int):
    name = label_mapping[label]
    return f'a picture of a {name}'

def caption_from_labels(labels_tensor: list):
    return [caption_from_label(label) for label in labels_tensor]


