"""Build the uploaded LViT architecture without importing global Config.py."""
from types import SimpleNamespace


MODEL_VERSION = 'uploaded_lvit_binary_224_v1'


def build_model():
    from src.lvit.nets.LViT import LViT
    transformer = SimpleNamespace(num_heads=4, num_layers=4, embeddings_dropout_rate=0.1,
                                  attention_dropout_rate=0.1, dropout_rate=0)
    config = SimpleNamespace(KV_size=960, transformer=transformer, expand_ratio=4,
                             patch_sizes=[16, 8, 4, 2], base_channel=64, n_classes=1)
    return LViT(config, n_channels=3, n_classes=1, img_size=224, vis=False)

