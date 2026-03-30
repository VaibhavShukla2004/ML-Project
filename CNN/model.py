import math

import torch
import torch.nn as nn
from torchvision import models


REGION_HEAD_NAMES = ('photo', 'name', 'expiry')

# Normalized ROI boxes derived from the synthetic template layout (width=1200, height=1600).
REGION_BOXES = {
    'photo': (75 / 1200, 250 / 1600, 320 / 1200, 625 / 1600),
    'name': (670 / 1200, 290 / 1600, 1180 / 1200, 390 / 1600),
    'expiry': (670 / 1200, 945 / 1600, 980 / 1200, 1030 / 1600),
}


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


def _make_projection_head(in_features, hidden_features, dropout_rate):
    return nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
    )


def _make_binary_head(in_features, dropout_rate):
    return nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate * 0.67),
        nn.Linear(128, 1),
    )


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
        projection_dim = 256

        self.backbone.classifier = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_projector = _make_projection_head(in_features, projection_dim, dropout_rate)
        self.region_projectors = nn.ModuleDict({
            region_name: _make_projection_head(in_features, projection_dim, dropout_rate)
            for region_name in REGION_HEAD_NAMES
        })
        self.region_heads = nn.ModuleDict({
            region_name: _make_binary_head(projection_dim, dropout_rate)
            for region_name in REGION_HEAD_NAMES
        })
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(projection_dim * (1 + len(REGION_HEAD_NAMES)) + len(REGION_HEAD_NAMES), 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.67),
            nn.Linear(256, num_classes),
        )
        self.head_modules = nn.ModuleList([
            self.global_projector,
            self.region_projectors,
            self.region_heads,
            self.fusion_head,
        ])

    def freeze_backbone(self, keep_stem_trainable=False):
        """Freeze backbone for Phase 1, optionally keeping the stem trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        for module in self.head_modules:
            for param in module.parameters():
                param.requires_grad = True

        # When using non-RGB inputs, let the stem adapt early to the new channels.
        if keep_stem_trainable:
            for param in self.backbone.features[0].parameters():
                param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning — Phase 2"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _extract_feature_map(self, x, freeze_backbone=False):
        if not freeze_backbone:
            return self.backbone.features(x)

        previous_mode = self.backbone.features.training
        self.backbone.features.eval()
        with torch.no_grad():
            feature_map = self.backbone.features(x)
        if previous_mode:
            self.backbone.features.train()
        return feature_map

    def _pool_region(self, feature_map, region_name):
        x1, y1, x2, y2 = REGION_BOXES[region_name]
        _, _, height, width = feature_map.shape

        left = min(width - 1, max(0, int(math.floor(x1 * width))))
        right = min(width, max(left + 1, int(math.ceil(x2 * width))))
        top = min(height - 1, max(0, int(math.floor(y1 * height))))
        bottom = min(height, max(top + 1, int(math.ceil(y2 * height))))

        pooled = self.global_pool(feature_map[:, :, top:bottom, left:right])
        return torch.flatten(pooled, 1)

    def _forward_from_feature_map(self, feature_map):
        global_features = torch.flatten(self.global_pool(feature_map), 1)
        global_embedding = self.global_projector(global_features)

        region_embeddings = []
        region_logits = []
        for region_name in REGION_HEAD_NAMES:
            region_features = self._pool_region(feature_map, region_name)
            region_embedding = self.region_projectors[region_name](region_features)
            region_logit = self.region_heads[region_name](region_embedding)
            region_embeddings.append(region_embedding)
            region_logits.append(region_logit)

        aux_logits = torch.cat(region_logits, dim=1)
        fusion_input = torch.cat([global_embedding, *region_embeddings, aux_logits], dim=1)
        fusion_logits = self.fusion_head(fusion_input)

        return {
            'fusion_logits': fusion_logits,
            'aux_logits': aux_logits,
        }

    def forward_head_with_frozen_features(self, x):
        """Forward for Phase 1 memory saving: frozen backbone features in no_grad, train heads only."""
        feature_map = self._extract_feature_map(x, freeze_backbone=True)
        return self._forward_from_feature_map(feature_map)

    def forward(self, x):
        feature_map = self._extract_feature_map(x, freeze_backbone=False)
        return self._forward_from_feature_map(feature_map)


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