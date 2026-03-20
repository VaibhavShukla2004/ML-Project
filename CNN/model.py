import torch
import torch.nn as nn
from torchvision import models


def _adapt_first_conv_for_in_channels(backbone, in_channels):
    """Adapt EfficientNet stem conv from RGB to custom input channels."""
    stem_conv = backbone.features[0][0]
    if not isinstance(stem_conv, nn.Conv2d):
        raise TypeError("Expected EfficientNet stem conv at features[0][0].")

    if stem_conv.in_channels == in_channels:
        return

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=stem_conv.out_channels,
        kernel_size=stem_conv.kernel_size,
        stride=stem_conv.stride,
        padding=stem_conv.padding,
        dilation=stem_conv.dilation,
        groups=stem_conv.groups,
        bias=(stem_conv.bias is not None),
    )

    with torch.no_grad():
        if in_channels > stem_conv.in_channels:
            new_conv.weight[:, :stem_conv.in_channels, :, :] = stem_conv.weight
            extra = in_channels - stem_conv.in_channels
            mean_rgb = stem_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, stem_conv.in_channels:, :, :] = mean_rgb.repeat(1, extra, 1, 1)
        else:
            new_conv.weight.copy_(stem_conv.weight[:, :in_channels, :, :])

        if stem_conv.bias is not None:
            new_conv.bias.copy_(stem_conv.bias)

    backbone.features[0][0] = new_conv


class DocumentForgeryDetector(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.3, in_channels=3):
        super(DocumentForgeryDetector, self).__init__()

        # Load pretrained EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        _adapt_first_conv_for_in_channels(self.backbone, in_channels=in_channels)

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

    def freeze_backbone(self, keep_stem_trainable=False):
        """Freeze backbone for Phase 1, optionally keeping the stem trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Always train the classification head in Phase 1.
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        # When using RGB+ELA, let the stem adapt early to the new channel.
        if keep_stem_trainable:
            for param in self.backbone.features[0].parameters():
                param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning — Phase 2"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward_head_with_frozen_features(self, x):
        """Forward for Phase 1 memory saving: frozen backbone features in no_grad, train head only.

        Use this only when the stem is also frozen. It avoids storing backbone activations,
        significantly reducing VRAM at high input resolution.
        """
        previous_mode = self.backbone.features.training
        self.backbone.features.eval()
        with torch.no_grad():
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        if previous_mode:
            self.backbone.features.train()
        return self.backbone.classifier(x)

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes=4, dropout_rate=0.3, in_channels=3):
    model = DocumentForgeryDetector(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        in_channels=in_channels,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on: {device}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    return model, device