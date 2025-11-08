import os
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt


IMAGENET_DIR = "./imagenet_samples"
IMAGENET_JSON = "./imagenet_class_index.json"
OUT_DIR = "./explanations"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet-18
try:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
except Exception:
    model = models.resnet18(pretrained=True)
model = model.to(device)
model.eval()

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Class index
with open(IMAGENET_JSON, "r") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]


def load_image(path: str) -> Tuple[torch.Tensor, Image.Image]:
    img = Image.open(path).convert("RGB")
    tensor = preprocess(img)
    return tensor, img


def predict_class(input_tensor: torch.Tensor) -> int:
    with torch.no_grad():
        logits = model(input_tensor.unsqueeze(0).to(device))
    _, idx = torch.max(logits, dim=1)
    return int(idx.item())


def make_grid_segments(h: int = 224, w: int = 224, patch_size: int = 28):
    """
    28x28 patches -> 8x8=64 segments for 224x224.
    Returns:
        segments: (H,W) numpy array with ids 0..K-1
        K: number of patches
    """
    seg = np.zeros((h, w), dtype=np.int64)
    k = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            seg[i:i + patch_size, j:j + patch_size] = k
            k += 1
    return seg, k


SEGMENTS, NUM_PATCHES = make_grid_segments()


def apply_mask(input_tensor: torch.Tensor,
               mask: np.ndarray,
               segments: np.ndarray) -> torch.Tensor:
    """
    input_tensor: (3,H,W), normalized
    mask: (K,) in {0,1}
    segments: (H,W) with values 0..K-1
    Returns masked tensor with 0-segments replaced by per-image mean color.
    """
    x = input_tensor.clone()
    # per-image mean color (in normalized space)
    baseline = x.mean(dim=(1, 2), keepdim=True)
    mask_2d = mask[segments]  # (H,W)
    mask_3d = torch.from_numpy(mask_2d).to(x.device).float().unsqueeze(0)
    x = x * mask_3d + baseline * (1.0 - mask_3d)
    return x


def lime_explanation(
    input_tensor: torch.Tensor,
    pred_idx: int,
    segments: np.ndarray,
    num_patches: int,
    num_samples: int = 1000,
    sigma: float = 0.25,
    lambda_reg: float = 1e-3,
) -> np.ndarray:
    """
    LIME-style linear surrogate on patch indicators for target logit.
    Returns:
        w: (K,) weights per patch.
    """
    K = num_patches

    # sample binary masks; include original as all-ones
    masks = np.random.binomial(1, 0.5, size=(num_samples, K)).astype(np.float32)
    masks[0, :] = 1.0

    # build batch of perturbed images
    imgs = []
    for m in masks:
        imgs.append(apply_mask(input_tensor, m, segments))
    batch = torch.stack(imgs, dim=0).to(device)  # (M,3,H,W)

    # model scores: use logits for the predicted class
    with torch.no_grad():
        logits = model(batch)          # (M,1000)
        scores = logits[:, pred_idx]   # (M,)
        y = scores.cpu().numpy()

    # locality weights: based on Hamming distance (#zeros)
    distances = np.sum(masks == 0, axis=1) / float(K)
    weights = np.exp(-(distances ** 2) / (sigma ** 2))

    X = masks                          # (M,K)
    W = np.diag(weights.astype(np.float32))  # (M,M)

    # weighted ridge regression: w = (X^T W X + λI)^(-1) X^T W y
    A = X.T @ W @ X + lambda_reg * np.eye(K, dtype=np.float32)
    b = X.T @ W @ y
    w = np.linalg.solve(A, b)          # (K,)

    return w


def lime_heatmap(weights: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Map patch weights -> per-pixel [0,1] heatmap for visualization.
    """
    h, w = segments.shape
    heat = np.zeros((h, w), dtype=np.float32)
    for k in range(weights.shape[0]):
        heat[segments == k] = weights[k]
    m, M = heat.min(), heat.max()
    if M > m:
        heat = (heat - m) / (M - m)
    return heat


def smoothgrad(
    input_tensor: torch.Tensor,
    pred_idx: int,
    stdev_spread: float = 0.15,
    n_samples: int = 50,
) -> np.ndarray:
    """
    SmoothGrad saliency map (H,W) for predicted class pred_idx.
    - noise std based on input range
    - saliency = sum_c |E[grad_c]|
    """
    x = input_tensor.unsqueeze(0).to(device)  # (1,3,H,W)
    x_min, x_max = x.min().item(), x.max().item()
    stdev = stdev_spread * (x_max - x_min)

    grads = []
    for _ in range(n_samples):
        noise = torch.normal(mean=0.0, std=stdev, size=x.shape).to(device)
        x_noisy = (x + noise).detach().clone().requires_grad_(True)

        logits = model(x_noisy)
        score = logits[0, pred_idx]  
        model.zero_grad(set_to_none=True)
        score.backward()

        grad = x_noisy.grad[0]       
        grads.append(grad)

    grads = torch.stack(grads, dim=0)    
    avg_grad = grads.mean(dim=0)         
    saliency = avg_grad.abs().sum(dim=0) 

    saliency = saliency - saliency.min()
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return saliency.cpu().numpy()


def aggregate_saliency_over_segments(
    saliency: np.ndarray,
    segments: np.ndarray,
    num_patches: int
) -> np.ndarray:
    scores = np.zeros(num_patches, dtype=np.float32)
    for k in range(num_patches):
        mask = (segments == k)
        if np.any(mask):
            scores[k] = saliency[mask].mean()
    return scores

def to_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks + 1.0  # 1..n


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    n = len(a)
    ra = to_ranks(a)
    rb = to_ranks(b)
    d = ra - rb
    return 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1))


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    n = len(a)
    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 or db == 0:
                continue
            s = np.sign(da * db)
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    if conc + disc == 0:
        return 0.0
    return (conc - disc) / (conc + disc)


def save_explanations(img_path, pil_img, lime_map, sg_map):
    """
    Figure with:
    - Original image
    - LIME heatmap overlay (jet)
    - SmoothGrad heatmap (jet)
    """
    plt.figure(figsize=(12, 4))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(pil_img)
    plt.axis("off")
    plt.title("Original")

    # LIME overlay
    plt.subplot(1, 3, 2)
    plt.imshow(pil_img)
    plt.imshow(lime_map, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.title("LIME (patch importance)")

    # SmoothGrad map
    plt.subplot(1, 3, 3)
    plt.imshow(sg_map, cmap="jet")
    plt.axis("off")
    plt.title("SmoothGrad saliency")

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUT_DIR, f"{base}_explanations.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved explanations to {out_path}")


def collect_image_paths(root: str, max_images: int = 5) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".jpeg", ".JPEG"}
    paths = []
    for fname in sorted(os.listdir(root)):
        full = os.path.join(root, fname)
        if os.path.isfile(full) and os.path.splitext(fname)[1] in exts:
            paths.append(full)
    return paths[:max_images]


def main():
    image_paths = collect_image_paths(IMAGENET_DIR, max_images=5)
    if not image_paths:
        print("No images found in ./imagenet_samples")
        return

    print("Found images:")
    for p in image_paths:
        print(" -", p)
    print()

    results = []

    for img_path in image_paths:
        input_tensor, pil_img = load_image(img_path)
        pred_idx = predict_class(input_tensor)
        pred_label = idx2label[pred_idx]
        pred_syn = idx2synset[pred_idx]

        print(f"\nImage: {img_path}")
        print(f"  Predicted: {pred_syn} ({pred_label})")

        # LIME
        lime_w = lime_explanation(
            input_tensor,
            pred_idx,
            SEGMENTS,
            NUM_PATCHES,
            num_samples=1000,
            sigma=0.25,
            lambda_reg=1e-3,
        )
        lime_map = lime_heatmap(lime_w, SEGMENTS)

        # SmoothGrad
        sg_map = smoothgrad(
            input_tensor,
            pred_idx,
            stdev_spread=0.15,
            n_samples=50,
        )

        sg_scores = aggregate_saliency_over_segments(
            sg_map,
            SEGMENTS,
            NUM_PATCHES,
        )

        rho = spearman_corr(lime_w, sg_scores)
        tau = kendall_tau(lime_w, sg_scores)

        print(f"  Spearman(LIME vs SmoothGrad) = {rho:.4f}")
        print(f"  KendallTau(LIME vs SmoothGrad) = {tau:.4f}")

        save_explanations(img_path, pil_img, lime_map, sg_map)

        results.append({
            "image": os.path.basename(img_path),
            "pred_synset": pred_syn,
            "pred_label": pred_label,
            "spearman": float(rho),
            "kendall": float(tau),
        })

    print("\nSummary:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
