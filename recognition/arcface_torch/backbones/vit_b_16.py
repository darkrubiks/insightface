import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class ViT(nn.Module):
    def __init__(self, num_classes: int = 512, dropout: float = 0.0) -> None:
        super(ViT, self).__init__()

        self.ViT = vit_b_16(dropout=dropout)
        self.ViT.heads = nn.Identity()
        self.embeddings = nn.Sequential(
            nn.Linear(768, num_classes, bias=False),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, x):
        feats = self.ViT(x)
        logits = self.embeddings(feats)

        return logits
    
if __name__ == "__main__":
    model = ViT(num_classes=512)
    print(model)
    model.eval()
    x = torch.randn(1, 3, 224, 224)  # Example input
    output = model(x)
    print(output.shape)  # Should be [1, 512]