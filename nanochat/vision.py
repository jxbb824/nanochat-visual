import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from torchvision import transforms as T
    _HAS_TORCHVISION = True
except Exception:
    # Torchvision is only needed for the optional patch-based encoder.
    T = None
    _HAS_TORCHVISION = False


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


class CLIPPatchVisionPrefixEncoder(nn.Module):
    """
    CLIP-based patch encoder that exposes patch-level tokens instead of a single
    global embedding. The underlying CLIP vision backbone is kept frozen and
    only the projection to the GPT embedding space is trained.
    """

    def __init__(
        self,
        d_model,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        pool: int = 1,
        device=None,
    ):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip_torch is required for CLIPPatchVisionPrefixEncoder. "
                "Install it with `pip install open_clip_torch`."
            ) from e

        self.pool = int(pool)
        assert self.pool >= 1, "pool must be >= 1"

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        visual = self.clip_model.visual
        if not hasattr(visual, "conv1") or not hasattr(visual, "transformer"):
            raise ValueError("CLIPPatchVisionPrefixEncoder currently only supports ViT-style CLIP backbones.")

        # Grid size of patch tokens before any pooling.
        grid_size = getattr(visual, "grid_size", None)
        if grid_size is None:
            image_size = getattr(visual, "image_size", 224)
            patch_size = getattr(visual, "patch_size", 32)
            if isinstance(image_size, (tuple, list)):
                image_size = image_size[0]
            if isinstance(patch_size, (tuple, list)):
                patch_size = patch_size[0]
            grid_h = image_size // patch_size
            grid_w = image_size // patch_size
        else:
            if isinstance(grid_size, (tuple, list)):
                grid_h, grid_w = grid_size
            else:
                grid_h = grid_w = int(grid_size)

        assert grid_h > 0 and grid_w > 0
        assert grid_h % self.pool == 0 and grid_w % self.pool == 0, "patch grid must be divisible by pool"

        self.grid_h = grid_h
        self.grid_w = grid_w
        pooled_h = grid_h // self.pool
        pooled_w = grid_w // self.pool
        self.num_tokens = pooled_h * pooled_w

        # Patch token dim before CLIP's projection head.
        visual_dim = getattr(visual, "width", None)
        if visual_dim is None:
            visual_dim = visual.conv1.out_channels
        self.proj = nn.Linear(visual_dim, d_model)

    def _encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run the frozen CLIP vision transformer and return patch tokens before
        the final pooling/projection.
        """
        visual = self.clip_model.visual

        # Adapted from CLIP ViT forward: we keep all tokens instead of just CLS.
        x = visual.conv1(images)  # (B, C, H', W')
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, C, H'*W')
        x = x.permute(0, 2, 1)  # (B, H'*W', C)

        class_embedding = visual.class_embedding.to(x.dtype)
        class_token = class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)  # (B, 1 + H'*W', C)

        # CLIP ViT stores positional_embedding as (N, C) or (1, N, C) depending
        # on the implementation. We follow the original forward: broadcast
        # positional embeddings along the batch dimension without resizing.
        pos_embed = visual.positional_embedding.to(dtype=x.dtype, device=x.device)
        if pos_embed.ndim == 2:
            # (N, C) -> broadcast to (B, N, C)
            assert (
                pos_embed.shape[0] == x.shape[1]
            ), f"Unexpected positional_embedding length: {pos_embed.shape[0]} vs sequence {x.shape[1]}"
            x = x + pos_embed  # broadcasting over batch dim
        elif pos_embed.ndim == 3:
            # (1, N, C) or (B, N, C)
            assert (
                pos_embed.shape[1] == x.shape[1]
            ), f"Unexpected positional_embedding length: {pos_embed.shape[1]} vs sequence {x.shape[1]}"
            x = x + pos_embed
        else:
            raise ValueError(f"Unsupported positional_embedding shape: {pos_embed.shape}")
        if hasattr(visual, "patch_dropout"):
            x = visual.patch_dropout(x)
        x = visual.ln_pre(x)

        attn_mask = getattr(visual, "attn_mask", None)
        if getattr(visual.transformer, "batch_first", True):
            x = visual.transformer(x, attn_mask=attn_mask)  # (B, N, C)
        else:
            x = x.permute(1, 0, 2)  # N, B, C
            x = visual.transformer(x, attn_mask=attn_mask)
            x = x.permute(1, 0, 2)  # B, N, C

        # Apply post layer norm (critical for stable features!)
        # CLIP ViT applies ln_post after transformer; without this the features
        # are unnormalized and have different distribution from pretraining.
        if hasattr(visual, "ln_post"):
            x = visual.ln_post(x)

        # Drop CLS token, keep only patch tokens.
        patch_tokens = x[:, 1:, :]  # (B, H'*W', C)
        return patch_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: Tensor of shape (B, 3, H, W), preprocessed with CLIP transforms.
        Returns:
        - prefix: Tensor of shape (B, num_tokens, d_model)
        """
        tokens = self._encode_patches(images)  # (B, N, C_clip)
        B, N, C_clip = tokens.shape

        # Reshape to 2D grid for optional spatial pooling.
        H = self.grid_h
        W = self.grid_w
        assert N == H * W, f"Unexpected number of patch tokens: got {N}, expected {H * W}"

        x = tokens.transpose(1, 2).reshape(B, C_clip, H, W)
        if self.pool > 1:
            x = F.avg_pool2d(x, kernel_size=self.pool, stride=self.pool)
            B, C_clip, H, W = x.shape

        x = x.reshape(B, C_clip, H * W).transpose(1, 2)  # (B, num_tokens, C_clip)
        x = self.proj(x)  # (B, num_tokens, d_model)
        return x


class PatchVisionPrefixEncoder(nn.Module):
    """
    Patch-based vision encoder that maps an image to a sequence of prefix tokens.
    Inspired by pixel-shuffle style spatial compression used in recent small VLMs.
    """

    def __init__(self, d_model, image_size=224, patch_size=16, pool=2):
        """
        Args:
            d_model: target embedding dim (must match GPT n_embd).
            image_size: square resize size for the input image.
            patch_size: patch size for the initial Conv2d patch embedding.
            pool: spatial pooling factor to reduce the number of visual tokens.
        """
        super().__init__()
        if not _HAS_TORCHVISION:
            raise ImportError(
                "torchvision is required for PatchVisionPrefixEncoder. "
                "Install it with `pip install torchvision`."
            )

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        grid_h = image_size // patch_size
        grid_w = image_size // patch_size
        assert grid_h % pool == 0 and grid_w % pool == 0, "patch grid must be divisible by pool"

        self.image_size = image_size
        self.patch_size = patch_size
        self.pool = pool

        # Simple patch embedding: 3xHxW -> (d_model, H', W') where H' = W' = image_size / patch_size.
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=False,
        )

        # After pooling, we get a smaller grid whose flattened size is the number of visual tokens.
        pooled_h = grid_h // pool
        pooled_w = grid_w // pool
        self.num_tokens = pooled_h * pooled_w

        # CLIP-like image preprocessing pipeline.
        self.preprocess = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def forward(self, images):
        """
        images: Tensor of shape (B, 3, H, W), already preprocessed.
        Returns:
        - prefix: Tensor of shape (B, num_tokens, d_model)
        """
        x = self.patch_embed(images)  # (B, d_model, H', W')
        if self.pool > 1:
            x = F.avg_pool2d(x, kernel_size=self.pool, stride=self.pool)  # spatial compression
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, num_tokens, d_model)
        return x

