import torch
import torch.nn as nn


class VisionPrefixEncoder(nn.Module):
    """
    Simple CNN-based vision encoder that maps a batch of images to a sequence of prefix tokens.
    Kept as a lightweight fallback.
    """

    def __init__(self, d_model, num_tokens=16):
        super().__init__()
        self.num_tokens = num_tokens
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((self.num_tokens, 1))

    def forward(self, images):
        """
        images: Tensor of shape (B, C, H, W), values in [0, 1] or [0, 255].
        Returns:
        - prefix: Tensor of shape (B, num_tokens, d_model)
        """
        if images.dtype != torch.float32:
            images = images.float()
        if images.max() > 1.5:
            images = images / 255.0
        x = self.cnn(images)  # (B, d_model, H', W')
        x = self.pool(x)      # (B, d_model, num_tokens, 1)
        x = x.squeeze(-1).transpose(1, 2)  # (B, num_tokens, d_model)
        return x


class CLIPVisionPrefixEncoder(nn.Module):
    """
    CLIP-based vision encoder.
    Uses a pretrained CLIP image encoder (e.g. ViT-B/32) and projects the global
    image embedding into a small sequence of prefix tokens in the GPT embedding space.
    """

    def __init__(self, d_model, num_tokens=16, model_name="ViT-B-32", pretrained="openai", device=None):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip_torch is required for CLIPVisionPrefixEncoder. "
                "Install it with `pip install open_clip_torch`."
            ) from e

        self.num_tokens = num_tokens
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        visual_dim = self.clip_model.visual.output_dim
        self.proj = nn.Linear(visual_dim, d_model * num_tokens)

    def forward(self, images):
        """
        images: Tensor of shape (B, 3, H, W), already preprocessed with CLIP transforms.
        Returns:
        - prefix: Tensor of shape (B, num_tokens, d_model)
        """
        feat = self.clip_model.encode_image(images)  # (B, visual_dim)
        x = self.proj(feat)  # (B, num_tokens * d_model)
        B = x.size(0)
        x = x.view(B, self.num_tokens, -1)
        return x

