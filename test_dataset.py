from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def main():
    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
    ])

    imagenet_s_50 = ImageFolder(
        root='datasets/ImageNet-S-50/train',
        transform=transform,
    )

    idx_to_class = {v: k for k, v in imagenet_s_50.class_to_idx.items()}

    loader = DataLoader(
        dataset=imagenet_s_50,
        batch_size=8,
        shuffle=True,
        pin_memory=True,
    )

    for image, label in loader:
        plt.imshow(image[0].permute(1, 2, 0))
        plt.title(idx_to_class[label[0].item()])
        plt.show()
        break

if __name__ == '__main__':
    main()