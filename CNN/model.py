import torch
import torch.nn as nn
from torchvision import models


class DocumentForgeryDetector(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super(DocumentForgeryDetector, self).__init__()

        # Load pretrained EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Get the number of features the backbone outputs
        classifier_layer = self.backbone.classifier[1]
        if not isinstance(classifier_layer, nn.Linear):
            raise TypeError("Expected EfficientNet classifier[1] to be nn.Linear")
        in_features = classifier_layer.in_features

        # Replace the default classifier with our custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.67),  # Slightly less dropout deeper in
            nn.Linear(256, num_classes)
        )

    def freeze_backbone(self):
        """Freeze all layers except the classifier head — Phase 1"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze only the head
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning — Phase 2"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes=4, dropout_rate=0.3):
    model = DocumentForgeryDetector(num_classes=num_classes, dropout_rate=dropout_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on: {device}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    return model, device