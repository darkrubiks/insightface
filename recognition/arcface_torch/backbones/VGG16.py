import torch
from torchvision.models import vgg16_bn, vgg16
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes: int = 512, dropout: float = 0.5) -> None:
        super(VGG16, self).__init__()
        self.vgg = vgg16_bn(dropout=dropout)
        self.vgg.classifier = nn.Identity()
        self.vgg.avgpool = nn.Identity()
        self.embeddings = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, num_classes, bias=False),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vgg(x)
        logits = self.embeddings(feats)

        return logits
    
if __name__ == "__main__":
    model = VGG16(num_classes=512)
    print(model)
    model.eval()
    x = torch.randn(1, 3, 224, 224)  # Example input
    output = model(x)
    print(output.shape)  # Should be [1, 512]