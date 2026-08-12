# DinoRADE: Paper → Code Walkthrough

A section-by-section map from the DinoRADE paper's Methodology to the modules that implement
it in this repository.

> **Reference paper**: [DinoRADE: Full Spectral Radar-Camera Fusion with Vision Foundation Model
> Features for Multi-class Object Detection in Adverse Weather](https://arxiv.org/abs/2604.08074),
> CVPR 2026 DriveX Workshop.
>
> DinoRADE builds directly on [RADE-Net](https://arxiv.org/abs/2602.19994) (IV 2026), which
> supplies the radar backbone, neck, detection head, and loss. Reading the RADE-Net paper first
> makes this document much shorter.

## Table of Contents

- [Orientation](#orientation)
- [The Input Tensor](#the-input-tensor)
- [§3.1 Radar Feature Extraction](#31-radar-feature-extraction--m_bevrad)
- [§3.2 Camera Feature Extraction](#32-camera-feature-extraction--m_vitup)
- [§3.3 Weighted Feature Lifting](#33-weighted-feature-lifting--m_q3d)
- [§3.4 Deformable Cross-Attention](#34-deformable-cross-attention--m_bevcam)
- [§3.5 Adaptive Fusion](#35-adaptive-fusion--m_f)
- [§3.6 Detection Head](#36-detection-head)
- [§3.7 Loss](#37-loss)
- [Ground Truth Encoding](#ground-truth-encoding)
- [Inference](#inference)
- [Shape Reference](#shape-reference)
- [Known Config Gaps](#known-config-gaps)

---

## Orientation

One repository, two papers. **RADE-Net** is the radar-only model: projection → UNet backbone →
dilated neck → centerpoint heads → focal + GWD + L1 loss. **DinoRADE** reuses all of that and
inserts a camera branch plus a fusion block between backbone and neck. So:

```
DinoRADE = RADE-Net + (DINOv3 branch) + (weighted lifting → cross-attention → gated fusion)
```

The whole model is a dict-passing pipeline: every module takes `batch_dict` and adds keys to it.
The DinoRADE forward path is [`model_assembly.py:95-141`](../models/model_assembly.py#L95-L141) —
45 lines, and every paper section maps to one of them:

```
batch_dict['rdr_era_dra']  →  pad 107→112
  camera:  PIL crop → DINOv3 → image_neck  → batch_dict['cam_features']
  radar:   rad_backbone                    → batch_dict['backbone_output']
  fusion   (lift + xattn + gate)           → overwrites 'backbone_output'
  rad_neck → head                          → 'heatmap', 'regression'
```

---

## The Input Tensor

*Paper Fig. 1, the "RAD ⊕ RAE projection" box.*

**Code**: [`data/create_projections.py:62-80`](../data/create_projections.py#L62-L80)

K-Radar ships a 4D tensor `arrDREA` of shape `(D=64, R=256, E=37, A=107)` per frame. The script
log-scales it, then collapses it two different ways:

| Line | Operation | Result |
| ---- | --------- | ------ |
| 68 | `max` over Doppler | RAE `(256, 107, 37)` |
| 69 | `max` over Elevation | DRA `(64, 256, 107)` |
| 70 | transpose RAE | ERA `(37, 256, 107)` |
| 72 | `concat` on channel axis | **`(101, 256, 107)`** |

That `101 = 64` Doppler bins `+ 37` elevation bins, stacked as *channels* over a range × azimuth
image. This is the core trick inherited from RADE-Net: a 4D tensor becomes a 2D image with 101
channels, so ordinary 2D CNNs apply and memory drops 91.9%. Saved as `DERA_tesseract_XXXXX.npy`.

> ⚠️ Line 90 of `create_projections.py` has a bare `break` inside the frame loop. As shipped, the
> script processes **one frame per subset**. The inline comment says to remove it; you must.

The dataloader ([`load_data.py:86-114`](../data/load_data.py#L86-L114)) simply `np.load`s that
file — no augmentation, no normalization at load time, since both were baked in at projection
time. [`app_utils.py:263`](../app_utils.py#L263) then wraps it as `batch_dict['rdr_era_dra']`,
shape **`(B, 101, 256, 107)`**.

Width 107 is not divisible by 8, so [`model_assembly.py:98`](../models/model_assembly.py#L98)
zero-pads it to **112**. Every "112" in the paper traces back to this one line.

---

## §3.1 Radar Feature Extraction → `M_BEV,Rad`

**Code**: [`models/backbone/unet.py:16-125`](../models/backbone/unet.py#L16-L125)

A standard depth-3 UNet with one twist: the skip connections pass through CBAM (channel +
spatial attention) before concatenating —
[`unet.py:94-119`](../models/backbone/unet.py#L94-L119).

Output `batch_dict['backbone_output']` = **`(B, 128, 256, 112)`**, the paper's
`M_BEV,Rad ∈ ℝ^(256×112×128)`. Each range-azimuth bin now carries a 128-dim feature in which
Doppler and elevation information has been encoded.

The paper's *"dual encoding architecture … both 3D projections undergo independent processing
before concatenation"* is [`InputStem`](../models/backbone/unet_blocks.py#L287-L322): it splits
channels `[:64]` (Doppler) and `[64:]` (elevation), applies a 1×1 conv to each separately, then
mixes. The shipped config sets `input_stem: False`, which falls back to a single
`Conv2d(101→128)` at [`unet.py:36`](../models/backbone/unet.py#L36). Set it to `True` to match
§3.1.

---

## §3.2 Camera Feature Extraction → `M_ViT,up`

**Image preparation**: inline in the forward,
[`model_assembly.py:100-123`](../models/model_assembly.py#L100-L123)

- `crop((0, 0, width//2, height))` — K-Radar's `cam-front` PNG is a **stereo pair side by side**;
  this keeps the left camera only.
- The processor resizes to `shortest_edge=720` and center-crops to 720×1280, the paper's stated
  resolution.

**Backbone**: [`fusion_assembly.py:52-77`](../models/fusion/fusion_assembly.py#L52-L77)

`AutoModel.from_pretrained` on a local DINOv3 ViT-S/16, then `.eval()` and `requires_grad=False`
on every parameter — frozen, as the paper states (21M frozen weights). Download it once with
[`utils/fusion/hugface_model.py`](../utils/fusion/hugface_model.py); the checkpoint is
`facebook/dinov3-vits16-pretrain-lvd1689m`.

720×1280 ÷ 16 = 45×80 = **3600 patches**, exactly the paper's number. The output
`last_hidden_state` is `(B, 3605, 384)` — 3600 patch tokens + 1 CLS + 4 register tokens.

**FPN**: [`CamNeck`](../models/fusion/radcam_nets.py#L79-L184)

Line 154 drops the 5 prefix tokens (`cam_features[:, 5:, :]`), a `Linear` projects 384→128, the
sequence is reshaped to `(B, 128, 45, 80)`, and two stride-2 `ConvTranspose2d` layers upsample to
**`(B, 128, 180, 320)`** = `M_ViT,up`. The paper's "two-layer FPN" ✓.

> The config ships `cam_neck_output: '256x112'`, which selects the *older* concat-fusion path.
> DinoRADE requires `'180x320'`.

---

## §3.3 Weighted Feature Lifting → `M_Q3D`

*Paper equations (1) and (2). This is the `+W` term in the Table 4 ablation (+1.77 AP₃D).*

**Code**: [`radcam_nets.py:628-655`](../models/fusion/radcam_nets.py#L628-L655)

```python
era = era_dra[:, 64:, :, :]                          # ← P^RAE, the 37 elevation channels
elevation_logits = self.weighting_network(era)       # CNN: 37 → 32 → 64 → 32 → 10
elevation_weights = torch.softmax(elevation_logits, dim=1)
...
query_features = query_features.unsqueeze(-1).repeat(1, 1, 1, 1, 10)   # (B,128,256,112,10)
query_features = query_features * elevation_weights                     # eq. (1)
```

**Why this matters.** The BEV map has no height dimension. To build 3D queries you copy it E=10
times along elevation — but a naive copy asserts "this object is equally likely at every height,"
which is wrong. So the **raw RAE projection** (the elevation half of the input tensor, never
touched by the backbone) is fed to a small CNN whose softmax over 10 bins says *where the radar
energy actually sits in elevation* for that range-azimuth cell. Multiply, and the 10 copies are
now weighted by real spectral evidence.

Result: **`(B, 128, 256, 112, 10)`** = `M_Q3D`.

The paper calls the weight predictor an "MLP"; the code's implementation is a 4-layer CNN
([`radcam_nets.py:511-532`](../models/fusion/radcam_nets.py#L511-L532)). Same role, and it is
selected by `DEFORM_ATTN_CFG['weighting_network'] = 'CNN'`.

---

## §3.4 Deformable Cross-Attention → `M_BEV,Cam`

Two halves: deciding *where to look* (reference points), then *looking* (deformable attention).

### Reference points — pure geometry, no learning

**Code**: [`get_reference_points()`](../models/fusion/radcam_nets.py#L534-L613)

For all 256 × 112 × 10 = **286,720** queries:

| Lines | Step |
| ----- | ---- |
| 543-548 | Build the polar grid at bin centers: r ∈ [0, 118] m, az ∈ [−53°, 53°], el ∈ [−18°, 18°] |
| 555-557 | Polar → Cartesian |
| 563-565 | Subtract the radar→camera extrinsic offsets (`RAD2CAM_CALIB`) |
| 575-580 | Apply `ldr2img` (rotation + translation from [`cam_front0.yml`](../calib/cam_calib/common/cam_front0.yml), parsed by [`camera_projection.py`](../utils/fusion/camera_projection.py)) |
| 587-593 | Apply intrinsics K, perspective divide → pixel (u, v) |
| 584, 597, 604 | Three validity masks: in front of camera, inside the left-camera crop, inside [0,1] after normalizing |

This is **Figure 2** of the paper — the red dot cloud projected onto the image. Every radar voxel
now knows which pixel it lands on.

### The attention

**Code**: [`radcam_nets.py:674-688`](../models/fusion/radcam_nets.py#L674-L688), calling into
[`MSDeformAttn`](../nets/ops/modules/ms_deform_attn.py#L79-L123)

The attention module is verbatim Deformable DETR (hence
[arXiv:2010.04159](https://arxiv.org/abs/2010.04159) being worth reading alongside). Internally:

```python
sampling_offsets   = Linear(query)            # "Offset prediction" box in Fig. 1
attention_weights  = softmax(Linear(query))
sampling_locations = reference_points + offsets / normalizer   # geometry + learned correction
output = bilinear_sample(value, sampling_locations) · attention_weights   # "Feature aggregation"
```

- **Queries**: 286,720 radar voxels (`M_Q3D` flattened).
- **Values**: 57,600 camera pixels (`M_ViT,up` flattened).

Each query samples `n_points` locations *near* its projected reference point and takes a learned
weighted sum. The geometry supplies the prior; the learned offsets absorb calibration error and
the fact that objects extend above their radar return.

`valid_mask` at [line 668](../models/fusion/radcam_nets.py#L668) zeros out queries that do not
project into the image, so they contribute nothing to the gradient.

**Elevation collapse** at [lines 685-688](../models/fusion/radcam_nets.py#L685-L688):
`torch.mean(fused_query, dim=4)` → back to **`(B, 128, 256, 112)`** = `M_BEV,Cam`. Paper: *"we
average the updated M_Q3D along the height dimension"* ✓. (An alternative learned
`Conv3d` reduction is available via `elevation_reduction: 'single_1d_conv'`.)

---

## §3.5 Adaptive Fusion → `M_f`

*Paper equations (3) and (4).*

**Code**: [`radcam_nets.py:697-703`](../models/fusion/radcam_nets.py#L697-L703)

```python
gate = self.fusion_gate(torch.cat([full_residual_query, fused_query], dim=-1))
fused_query = gate * fused_query + (1 - gate) * full_residual_query
```

`fusion_gate` is `Linear(256→64) → ReLU → Linear(64→1) → Sigmoid`
([lines 462-468](../models/fusion/radcam_nets.py#L462-L468)) — the paper's "MLP with two layers."
Per range-azimuth bin, the network decides how much camera evidence to trust.

Paper §5.1 explains why this matters: roughly 17k of K-Radar's 35k frames have a partly to fully
occluded camera (fog on the lens, glare, snow). The gate lets those frames fall back to radar
instead of being poisoned by a useless image.

**Two notational points when reading paper and code side by side:**

1. The code's `gate` is the **camera** weight; the paper's `Γ` is the **radar** weight. Same
   model, flipped convention.
2. The paper writes `Γ ∈ ℝ^(256×112×128)` (per-channel), but the code's final `Linear` outputs
   **1** — a scalar gate per bin, broadcast across channels.

---

## §3.6 Detection Head

**Code**: [`model_assembly.py:138-139`](../models/model_assembly.py#L138-L139) →
[`MultiChannelHeatmap`](../models/head/coupled_heads.py#L19-L78)

The fused map first passes through `rad_neck`
([`DilatedResidualNeck`](../models/neck/neck_nets.py#L154-L163) — three residual blocks with
dilation 1, 2, 3, widening the receptive field over neighbouring RA bins), then into two parallel
head branches sharing the same input:

| Branch | Structure | Output |
| ------ | --------- | ------ |
| **heatmap** | 3 × (Conv3×3 + GroupNorm + SiLU) + Conv1×1, sigmoid | `(B, 5, 256, 112)` — one channel per class, still in range-azimuth coordinates |
| **regression** | same trunk, 8 output channels | `(B, 8, 256, 112)` — `[dx, dy, dz, l, w, h, sin θ, cos θ]` |

Two design notes: `dx, dy` are sub-bin offsets (the bin index gives coarse position, these
correct it to sub-bin resolution); and `sin θ / cos θ` rather than `θ` avoids the wraparound
discontinuity at ±π.

---

## §3.7 Loss

**Code**: [`ops/loss_utils.py:124-160`](../ops/loss_utils.py#L124-L160)

```python
loss += 2 * (loss_heatmap   / loss_heatmap.detach().mean())
loss += gwd_loss            / gwd_loss.detach().mean()
loss += smooth_l1_loss      / smooth_l1_loss.detach().mean()
```

That is equation (17), `L_all = 2·L_foc + L_gwd + L_L1`, with each term divided by its own
detached mini-batch mean so no component dominates.

### Focal loss

Target built by
[`generate_gaussian_heatmap`](../ops/loss_utils.py#L8-L81): a Gaussian of width σ painted at each
GT center, in the *correct class channel*. The loss itself is CornerNet-style
[`FocalLossGaussianContinuous`](../ops/loss.py#L43-L79) with α=2, γ=4.

**This is where the σ = 0.75 finding lives.** Paper §3.7 argues that σ = 3 covers a physically
enormous area at long range, because the bins are polar — which drowns pedestrians and cyclists.
The shipped config has `SIGMA = 3` ([`params_radcam.py:104`](../configs/params_radcam.py#L104)),
which is RADE-Net's value. Set it to `0.75` to reproduce DinoRADE.

### GWD loss

[`GWDLoss.forward`](../ops/loss.py#L124-L167). Note it is **not** dense: it loops over ground
truth centers and reads the regression vector *at that exact bin*
(`regression_map[:, int(center_y), int(center_x)]`).
[`build_transformed_box`](../ops/loss.py#L169-L203) turns `(range_idx, azi_idx)` plus the
predicted offsets into an absolute Cartesian box, then both boxes are compared as 2D Gaussians —
fully differentiable, with no rotated-IoU polygon clipping.

### Smooth L1

[`ops/loss.py:206-226`](../ops/loss.py#L206-L226), applied to the same matched pairs. Paper §3.7
explains the motivation: GWD's gradient with respect to the covariance is weak once centers
align, so predicted boxes came out consistently smaller than the ground truth.

---

## Ground Truth Encoding

**Code**: [`data/ground_truth.py:201-243`](../data/ground_truth.py#L201-L243)

`build_gt_center_points` is the exact inverse of the head's coordinate transform. Given a ground
truth Cartesian `(x, y)`:

```
azi = -arctan2(y, x)         →  degrees, then + 53 to shift into [0°, 106°]
rng = sqrt(x² + y²)
azi_idx = azi · 111 / 106     →  [0, 111]   (112 bins, padded)
rng_idx = rng · 255 / 118     →  [0, 255]
```

If you ever debug a coordinate bug, this function and
[`transform_ra_indices_to_cartesian`](../ops/transformation.py#L32-L52) must remain exact
inverses of each other.

---

## Inference

**Code**: [`postprocessing`](../ops/nms.py#L202-L207) — three stages:

1. **Peak picking** — 3×3 max-pool NMS on the heatmap plus a confidence threshold (default 0.3).
2. **Box construction** — per-pixel `argmax` over the 5 class channels, RA → Cartesian, add the
   regression offsets, then filter by the wide ROI.
3. **Rotated NMS** — IoU threshold 0.3.

`MODE = 'test'` writes per-frame predictions as `.pkl` and only needs to run once;
`MODE = 'evaluate'` reuses those files to compute mAP.

---

## Shape Reference

End-to-end tensor shapes for a batch of size `B`:

| Stage | Key | Shape |
| ----- | --- | ----- |
| Raw K-Radar tesseract | `arrDREA` | `(64, 256, 37, 107)` |
| After projection | `DERA_tesseract_*.npy` | `(101, 256, 107)` |
| Batched input | `rdr_era_dra` | `(B, 101, 256, 107)` |
| After padding | `rdr_era_dra` | `(B, 101, 256, 112)` |
| Radar backbone (`M_BEV,Rad`) | `backbone_output` | `(B, 128, 256, 112)` |
| DINOv3 tokens (`M_ViT`) | `cam_features` | `(B, 3605, 384)` |
| Camera FPN (`M_ViT,up`) | `cam_features` | `(B, 128, 180, 320)` |
| Lifted 3D queries (`M_Q3D`) | — | `(B, 128, 256, 112, 10)` |
| After cross-attention (`M_BEV,Cam`) | — | `(B, 128, 256, 112)` |
| After gated fusion (`M_f`) | `backbone_output` | `(B, 128, 256, 112)` |
| Head — classification | `heatmap` | `(B, 5, 256, 112)` |
| Head — regression | `regression` | `(B, 8, 256, 112)` |

---

## Known Config Gaps

**The shipped configs will not run DinoRADE as-is.** `configs/params_radcam.py` is byte-identical
to `configs/params_rade.py`, and both are radar-only configurations. The fusion module reads
several keys that are absent:

| Location | Required | Shipped |
| -------- | -------- | ------- |
| [`params_radcam.py:20`](../configs/params_radcam.py#L20) | `APPROACH = 'RadarCamera'` | `'RadarOnly'` |
| [`radcam_nets.py:429-433`](../models/fusion/radcam_nets.py#L429-L433) | `DEFORM_ATTN_CFG` keys `expand_to_256`, `fusion_strategy`, `elevation_reduction`, `positional_embedding`, `weighting_network` | **absent** → `KeyError` |
| [`model_assembly.py:54`](../models/model_assembly.py#L54) | `CAM_CONFIG['double_size']` | **absent** → `KeyError` |
| [`load_data.py:80-81`](../data/load_data.py#L80-L81) | `cfg.LDR_COL_READ`, `cfg.LDR_PROCESSING_METHOD` | **absent** → `AttributeError` |
| [`params_radcam.py:180`](../configs/params_radcam.py#L180) | `cam_neck_output = '180x320'` | `'256x112'` |
| [`params_radcam.py:104`](../configs/params_radcam.py#L104) | `SIGMA = 0.75` | `3` |
| [`unet.py:36`](../models/backbone/unet.py#L36) | `UNET_CFG['input_stem'] = True` (paper §3.1) | `False` |

To match the paper, the fusion config should be `fusion_strategy='gated_fusion'`,
`elevation_reduction='mean'`, `weighting_network='CNN'`, and `expand_to_256=False` (the head
expects 128 input channels).

### Smaller discrepancies

- `DEFORM_ATTN_CFG['n_points']` is `8` in the config, while §3.4 states *"we predict four offsets
  around each reference point."*
- The README instructs building `deform_attn/`, while the code imports from `nets/ops/`. This is
  harmless — both `setup.py` files install the same C extension name
  `MultiScaleDeformableAttention` — but the Python wrapper actually imported is
  [`nets/ops/modules/ms_deform_attn.py`](../nets/ops/modules/ms_deform_attn.py).
- [`classification_loss`](../ops/loss_utils.py#L114) calls `ClassificationLoss(cfg, device)` while
  the class signature is [`__init__(self, device)`](../ops/loss.py#L230). Only reachable when
  `USE_SINGULAR_HEATMAP = True`, which is not the DinoRADE path.

None of this is hard to fix; it reads as though the public config was reverted to the radar-only
baseline before release. But it does mean a clone-and-run will not reproduce Table 2.
