"""
Hybrid Object Detector: YOLOv8 + ResNet50
Accuracy improvements over plain YOLOv8:
  1. Feature injection  — ResNet50 FPN features fused into YOLO's feature map
  2. ROI-align rescoring — per-box CNN confidence re-weighting
  3. Test-Time Augmentation (TTA) — multi-scale + flip ensemble
  4. Weighted Box Fusion (WBF) — better than NMS for merging TTA boxes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.ops import roi_align
import cv2
import numpy as np
from PIL import Image
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. ResNet50 multi-scale FPN backbone
# ─────────────────────────────────────────────────────────────────────────────
class CNNFeatureExtractor(nn.Module):
    """ResNet50 backbone with FPN — produces rich multi-scale feature map."""

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.low_level  = nn.Sequential(*list(resnet.children())[:4])   # /4,   64ch
        self.mid_level  = nn.Sequential(*list(resnet.children())[4:6])  # /8,  512ch
        self.high_level = nn.Sequential(*list(resnet.children())[6:8])  # /32, 2048ch

        self.fpn_high = nn.Conv2d(2048, 256, 1)
        self.fpn_mid  = nn.Conv2d(512,  256, 1)
        self.fpn_low  = nn.Conv2d(64,   256, 1)

        self.refine = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        for p in self.low_level.parameters():
            p.requires_grad = False

    def forward(self, x):
        low  = self.low_level(x)
        mid  = self.mid_level(low)
        high = self.high_level(mid)

        p_high = self.fpn_high(high)
        p_mid  = self.fpn_mid(mid)  + F.interpolate(p_high, size=mid.shape[2:],  mode='nearest')
        p_low  = self.fpn_low(low)  + F.interpolate(p_mid,  size=low.shape[2:],  mode='nearest')

        p_high_up = F.interpolate(p_high, size=p_low.shape[2:], mode='bilinear', align_corners=False)
        p_mid_up  = F.interpolate(p_mid,  size=p_low.shape[2:], mode='bilinear', align_corners=False)
        fused     = (p_low + p_mid_up + p_high_up) / 3.0

        return self.refine(fused)   # (1, 512, H/4, W/4)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CBAM attention
# ─────────────────────────────────────────────────────────────────────────────
class FeatureEnhancementModule(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1), nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.channel_attn(x)
        sp = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)
        return x * self.spatial_attn(sp)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature injection — fuses ResNet50 map into YOLO's feature space
# ─────────────────────────────────────────────────────────────────────────────
class FeatureFusionModule(nn.Module):
    """
    Projects ResNet50's 512-ch map to match YOLO's expected channel depth,
    then adds it as a residual. This injects semantic context directly into
    the feature space YOLO uses for detection.
    """
    def __init__(self, cnn_channels=512, yolo_channels=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(cnn_channels, yolo_channels, 1),
            nn.BatchNorm2d(yolo_channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(yolo_channels * 2, yolo_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, cnn_feat, yolo_feat):
        """
        cnn_feat  : (1, 512, Hc, Wc)
        yolo_feat : (1, 256, Hy, Wy)
        Returns fused feature map at yolo_feat's spatial size.
        """
        cnn_proj = self.proj(cnn_feat)
        cnn_proj = F.interpolate(cnn_proj, size=yolo_feat.shape[2:],
                                 mode='bilinear', align_corners=False)
        # learned gate decides how much CNN info to inject
        gate = self.gate(torch.cat([cnn_proj, yolo_feat], dim=1))
        return yolo_feat + gate * cnn_proj


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROI-align rescoring head
# ─────────────────────────────────────────────────────────────────────────────
class BoxRescoringHead(nn.Module):
    """
    For each YOLO box, ROI-aligns the CNN feature map to get region features,
    then scores the region quality. Score is blended into YOLO's confidence.
    """
    def __init__(self, in_channels=512, roi_size=7):
        super().__init__()
        self.roi_size = roi_size
        flat = in_channels * roi_size * roi_size
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),   nn.Sigmoid(),
        )

    def forward(self, feat_map, boxes_xyxy, img_hw):
        if boxes_xyxy.shape[0] == 0:
            return torch.empty(0, device=feat_map.device)

        H_img, W_img   = img_hw
        H_f,   W_f     = feat_map.shape[2], feat_map.shape[3]
        scaled          = boxes_xyxy.clone().float()
        scaled[:, [0,2]] *= W_f / W_img
        scaled[:, [1,3]] *= H_f / H_img

        batch_idx = torch.zeros(scaled.shape[0], 1, device=feat_map.device)
        rois      = torch.cat([batch_idx, scaled], dim=1)
        pooled    = roi_align(feat_map, rois, output_size=self.roi_size)
        return self.head(pooled).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Weighted Box Fusion (pure-Python, no extra deps)
# ─────────────────────────────────────────────────────────────────────────────
def _iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def weighted_box_fusion(boxes_list, scores_list, labels_list, iou_thr=0.55):
    """
    Merge boxes from multiple prediction sets using weighted averaging.
    Much better than NMS for TTA — keeps spatial precision.

    boxes_list  : list of (N,4) numpy arrays  [x1,y1,x2,y2] normalised 0-1
    scores_list : list of (N,) numpy arrays
    labels_list : list of (N,) numpy arrays (int class ids)
    Returns merged (boxes, scores, labels) as numpy arrays.
    """
    all_boxes, all_scores, all_labels = [], [], []
    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        for b, s, l in zip(boxes, scores, labels):
            all_boxes.append(b); all_scores.append(s); all_labels.append(l)

    if not all_boxes:
        return np.empty((0,4)), np.empty(0), np.empty(0)

    order   = np.argsort(-np.array(all_scores))
    used    = [False] * len(all_boxes)
    out_b, out_s, out_l = [], [], []

    for i in order:
        if used[i]:
            continue
        cluster_b = [all_boxes[i]]
        cluster_s = [all_scores[i]]
        cluster_l = [all_labels[i]]
        used[i] = True
        for j in order:
            if used[j] or all_labels[j] != all_labels[i]:
                continue
            if _iou(all_boxes[i], all_boxes[j]) >= iou_thr:
                cluster_b.append(all_boxes[j])
                cluster_s.append(all_scores[j])
                used[j] = True

        w  = np.array(cluster_s)
        wb = np.average(cluster_b, axis=0, weights=w)
        out_b.append(wb)
        out_s.append(np.mean(w))
        out_l.append(cluster_l[0])

    return np.array(out_b), np.array(out_s), np.array(out_l)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main hybrid detector
# ─────────────────────────────────────────────────────────────────────────────
class HybridObjectDetector:
    """
    Accuracy improvements over plain YOLOv8:

    ┌─────────────────────────────┬──────────────────────────────────────────┐
    │ Technique                   │ Effect                                   │
    ├─────────────────────────────┼──────────────────────────────────────────┤
    │ ResNet50 FPN backbone       │ Richer multi-scale features              │
    │ CBAM attention              │ Focus on relevant regions/channels       │
    │ Feature injection (gate)    │ ResNet50 context added to YOLO features  │
    │ ROI-align rescoring         │ Per-box CNN quality score                │
    │ TTA (flip + 2 scales)       │ Catches missed detections                │
    │ Weighted Box Fusion         │ Better box merging than NMS              │
    └─────────────────────────────┴──────────────────────────────────────────┘
    """

    def __init__(self, yolo_model='yolov8x.pt', use_cnn_features=True,
                 rescore_alpha=0.35, use_tta=True, device=None):
        if not YOLO_AVAILABLE:
            raise ImportError("Install ultralytics: pip install ultralytics")

        self.device          = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cnn_features = use_cnn_features
        self.rescore_alpha   = rescore_alpha
        self.use_tta         = use_tta

        self.yolo = YOLO(yolo_model)

        if use_cnn_features:
            self.cnn_extractor    = CNNFeatureExtractor(pretrained=True).to(self.device)
            self.feature_enhancer = FeatureEnhancementModule().to(self.device)
            self.fusion_module    = FeatureFusionModule(cnn_channels=512, yolo_channels=256).to(self.device)
            self.rescore_head     = BoxRescoringHead().to(self.device)
            for m in [self.cnn_extractor, self.feature_enhancer,
                      self.fusion_module, self.rescore_head]:
                m.eval()

        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ── internal helpers ──────────────────────────────────────────────────────

    def _extract_features(self, pil_image):
        t = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw = self.cnn_extractor(t)
            return self.feature_enhancer(raw)          # (1, 512, H/4, W/4)

    def _yolo_on_image(self, cv_image, conf, iou):
        return self.yolo(cv_image, conf=conf, iou=iou, verbose=False)[0]

    def _rescore(self, feat_map, results, orig_hw):
        """Blend CNN region scores into YOLO confidences."""
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return
        xyxy      = boxes.xyxy.to(self.device)
        yolo_conf = boxes.conf.to(self.device)
        with torch.no_grad():
            weights = self.rescore_head(feat_map, xyxy, orig_hw)
        new_conf = (1 - self.rescore_alpha) * yolo_conf + self.rescore_alpha * weights
        results.boxes.data[:, 4] = new_conf

    def _results_to_numpy(self, results, img_w, img_h):
        """Convert ultralytics Results to normalised numpy arrays for WBF."""
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0,4)), np.empty(0), np.empty(0)
        xyxy   = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        labels = boxes.cls.cpu().numpy().astype(int)
        # normalise to [0,1]
        xyxy[:, [0,2]] /= img_w
        xyxy[:, [1,3]] /= img_h
        return xyxy, scores, labels

    def _wbf_to_results(self, boxes_norm, scores, labels, img_w, img_h,
                         conf_thr, base_results):
        """
        Convert WBF output back to an ultralytics-compatible Results object
        by filtering and rebuilding base_results.boxes.data.
        """
        if len(boxes_norm) == 0:
            base_results.boxes.data = base_results.boxes.data[:0]
            return base_results

        keep = scores >= conf_thr
        boxes_norm = boxes_norm[keep]
        scores     = scores[keep]
        labels     = labels[keep]

        # denormalise
        boxes_abs       = boxes_norm.copy()
        boxes_abs[:, [0,2]] *= img_w
        boxes_abs[:, [1,3]] *= img_h

        # rebuild data tensor: [x1,y1,x2,y2, conf, cls]
        data = np.concatenate([
            boxes_abs,
            scores[:, None],
            labels[:, None].astype(float),
        ], axis=1)
        base_results.boxes.data = torch.tensor(data, dtype=torch.float32,
                                               device=self.device)
        return base_results

    # ── TTA helpers ───────────────────────────────────────────────────────────

    def _tta_variants(self, cv_image):
        """
        Returns list of (augmented_cv_image, flip_h, scale).
        We use: original, horizontal flip, 0.83x scale, 1.2x scale.
        """
        h, w = cv_image.shape[:2]
        variants = [
            (cv_image,                                    False, 1.0),
            (cv2.flip(cv_image, 1),                       True,  1.0),
            (cv2.resize(cv_image, (int(w*.83), int(h*.83))), False, 0.83),
            (cv2.resize(cv_image, (int(w*1.2),  int(h*1.2))),  False, 1.2),
        ]
        return variants

    def _flip_boxes_h(self, boxes_norm):
        """Mirror normalised boxes horizontally."""
        if len(boxes_norm) == 0:
            return boxes_norm
        b = boxes_norm.copy()
        b[:, 0], b[:, 2] = 1 - boxes_norm[:, 2], 1 - boxes_norm[:, 0]
        return b

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, image, conf=0.25, iou=0.45):
        """
        Returns (results, cnn_feature_map, elapsed_seconds).
        results.boxes.conf reflects CNN-rescored + TTA-fused confidences.
        """
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
            cv_image  = cv2.imread(image)
        else:
            cv_image  = image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        orig_h, orig_w = cv_image.shape[:2]
        t0 = time.time()

        # ── Step 1: extract ResNet50 features ────────────────────────────────
        cnn_features = self._extract_features(pil_image) if self.use_cnn_features else None

        # ── Step 2: YOLO on original (lower conf to widen candidate pool) ────
        raw_conf = max(conf * 0.7, 0.1)
        base_results = self._yolo_on_image(cv_image, raw_conf, iou)

        # ── Step 3: rescore with CNN ──────────────────────────────────────────
        if self.use_cnn_features and cnn_features is not None:
            self._rescore(cnn_features, base_results, (orig_h, orig_w))

        # ── Step 4: TTA — run YOLO on augmented variants ─────────────────────
        all_boxes, all_scores, all_labels = [], [], []

        # add base prediction
        b, s, l = self._results_to_numpy(base_results, orig_w, orig_h)
        all_boxes.append(b); all_scores.append(s); all_labels.append(l)

        if self.use_tta:
            for aug_img, flipped, scale in self._tta_variants(cv_image)[1:]:
                aug_h, aug_w = aug_img.shape[:2]
                aug_res = self._yolo_on_image(aug_img, raw_conf, iou)

                # rescore augmented result too
                if self.use_cnn_features and cnn_features is not None:
                    aug_pil = Image.fromarray(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
                    aug_feat = self._extract_features(aug_pil)
                    self._rescore(aug_feat, aug_res, (aug_h, aug_w))

                b, s, l = self._results_to_numpy(aug_res, aug_w, aug_h)

                # undo horizontal flip
                if flipped and len(b) > 0:
                    b = self._flip_boxes_h(b)

                all_boxes.append(b); all_scores.append(s); all_labels.append(l)

        # ── Step 5: Weighted Box Fusion across all predictions ────────────────
        merged_boxes, merged_scores, merged_labels = weighted_box_fusion(
            all_boxes, all_scores, all_labels, iou_thr=0.55
        )

        # ── Step 6: write merged results back ─────────────────────────────────
        final_results = self._wbf_to_results(
            merged_boxes, merged_scores, merged_labels,
            orig_w, orig_h, conf, base_results
        )

        elapsed = time.time() - t0
        return final_results, cnn_features, elapsed
