from torchvision.models import resnet50
from sentence_transformers import SentenceTransformer

caption_encoder = SentenceTransformer("all-MiniLM-L6-v2")
caption_encoder.eval()

image_encoder = resnet50()
