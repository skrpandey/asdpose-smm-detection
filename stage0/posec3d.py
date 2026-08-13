"""Standalone reimplementation of the ASDMotion PoseConv3D model and its
mmaction 0.24 test pipeline, for the Gate 0.3 baseline reproduction.

Ported from the checkpoint's embedded config and the TalBarami/mmaction2 fork:
ResNet3dSlowOnly(depth=50, in_channels=17, base_channels=32, num_stages=3,
stage_blocks=(3,4,6), conv1_kernel=(1,7,7), all strides per config) plus
I3DHead(num_classes=2). The test pipeline is UniformSampleFrames(48, 10,
test_mode, seed=255) -> PoseDecode -> PoseCompact(0.25, 10, hw_ratio=1,
allow_imgpad) -> Resize((-1,56)) -> CenterCrop(56) -> GeneratePoseTarget(
sigma=0.6, use_score, with_kp, double) -> NCTHW. Score = softmax mean over
the 20 clips (10 sampled x 2 for the horizontal flip copy).

Weights load with strict=True, which pins the architecture to the checkpoint.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LEFT_KP = (1, 3, 5, 7, 9, 11, 13, 15)
RIGHT_KP = (2, 4, 6, 8, 10, 12, 14, 16)


class ConvBN(nn.Module):
    """conv + bn (+ relu), submodules named to match mmcv ConvModule keys."""

    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, act=True):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x) if self.act is not None else x


class Bottleneck3d(nn.Module):
    """pytorch-style bottleneck; inflate_style 3x1x1."""

    expansion = 4

    def __init__(self, inplanes, planes, spatial_stride=1, temporal_stride=1,
                 downsample=None, inflate=True):
        super().__init__()
        if inflate:
            k1, p1 = (3, 1, 1), (1, 0, 0)
        else:
            k1, p1 = (1, 1, 1), (0, 0, 0)
        self.conv1 = ConvBN(inplanes, planes, k1, stride=1, padding=p1)
        self.conv2 = ConvBN(planes, planes, (1, 3, 3),
                            stride=(temporal_stride, spatial_stride, spatial_stride),
                            padding=(0, 1, 1))
        self.conv3 = ConvBN(planes, planes * self.expansion, 1, act=False)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv3(self.conv2(self.conv1(x)))
        return self.relu(out + identity)


class PoseConv3D(nn.Module):
    """ResNet3dSlowOnly backbone + I3DHead, exactly the checkpoint config."""

    def __init__(self, in_channels=17, base_channels=32,
                 stage_blocks=(3, 4, 6), spatial_strides=(2, 2, 2),
                 temporal_strides=(1, 1, 2), inflates=(0, 1, 1),
                 num_classes=2):
        super().__init__()
        self.conv1 = ConvBN(in_channels, base_channels, (1, 7, 7),
                            stride=(1, 1, 1), padding=(0, 3, 3))
        self.maxpool = nn.MaxPool3d((1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        inplanes = base_channels
        layers = []
        for i, blocks in enumerate(stage_blocks):
            planes = base_channels * 2 ** i
            ss, ts, inf = spatial_strides[i], temporal_strides[i], inflates[i]
            downsample = ConvBN(inplanes, planes * 4, 1,
                                stride=(ts, ss, ss), act=False)
            stage = [Bottleneck3d(inplanes, planes, ss, ts, downsample, bool(inf))]
            inplanes = planes * 4
            stage += [Bottleneck3d(inplanes, planes, inflate=bool(inf))
                      for _ in range(1, blocks)]
            layers.append(nn.Sequential(*stage))
        self.layer1, self.layer2, self.layer3 = layers

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_cls = nn.Linear(inplanes, num_classes)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.avg_pool(x).flatten(1)
        return self.fc_cls(x)

    def load_mmaction_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        remapped = {}
        for k, v in sd.items():
            k = k.replace("backbone.", "").replace("cls_head.", "")
            remapped[k] = v
        self.load_state_dict(remapped, strict=True)
        return ckpt.get("meta", {})


# ---------------------------------------------------------------------------
# test pipeline (mirrors mmaction 0.24 transforms, keypoint branch only)
# ---------------------------------------------------------------------------

def uniform_sample_frames_test(num_frames, clip_len=48, num_clips=10, seed=255):
    """UniformSampleFrames._get_test_clips, verbatim logic."""
    rng = np.random.RandomState(seed)
    if num_frames < clip_len:
        if num_frames < num_clips:
            start_inds = list(range(num_clips))
        else:
            start_inds = [i * num_frames // num_clips for i in range(num_clips)]
        inds = np.concatenate([np.arange(i, i + clip_len) for i in start_inds])
    elif clip_len <= num_frames < clip_len * 2:
        all_inds = []
        for _ in range(num_clips):
            basic = np.arange(clip_len)
            chosen = rng.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[chosen] = 1
            offset = np.cumsum(offset)
            all_inds.append(basic + offset[:-1])
        inds = np.concatenate(all_inds)
    else:
        bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
        bsize = np.diff(bids)
        bst = bids[:clip_len]
        all_inds = [bst + rng.randint(bsize) for _ in range(num_clips)]
        inds = np.concatenate(all_inds)
    return np.mod(inds, num_frames)


def pose_compact(kp, img_shape, padding=0.25, threshold=10, hw_ratio=(1.0, 1.0)):
    """PoseCompact with allow_imgpad=True. Modifies nothing; returns new kp, shape."""
    kp = kp.copy()
    h, w = img_shape
    kp[np.isnan(kp)] = 0.0
    kp_x, kp_y = kp[..., 0], kp[..., 1]
    min_x = np.min(kp_x[kp_x != 0], initial=np.inf)
    min_y = np.min(kp_y[kp_y != 0], initial=np.inf)
    max_x = np.max(kp_x[kp_x != 0], initial=-np.inf)
    max_y = np.max(kp_y[kp_y != 0], initial=-np.inf)
    if max_x - min_x < threshold or max_y - min_y < threshold:
        return kp, img_shape
    center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
    half_width = (max_x - min_x) / 2 * (1 + padding)
    half_height = (max_y - min_y) / 2 * (1 + padding)
    half_height = max(hw_ratio[0] * half_width, half_height)
    half_width = max(1 / hw_ratio[1] * half_height, half_width)
    min_x, max_x = int(center[0] - half_width), int(center[0] + half_width)
    min_y, max_y = int(center[1] - half_height), int(center[1] + half_height)
    kp_x[kp_x != 0] -= min_x
    kp_y[kp_y != 0] -= min_y
    return kp, (max_y - min_y, max_x - min_x)


def resize_short_edge(kp, img_shape, short=56):
    """Resize(scale=(-1, short), keep_ratio=True), mmcv.rescale_size rounding."""
    h, w = img_shape
    f = min(np.inf / max(h, w), short / min(h, w))
    new_w, new_h = int(w * f + 0.5), int(h * f + 0.5)
    factor = np.array([new_w / w, new_h / h], dtype=np.float32)
    return kp * factor, (new_h, new_w)


def center_crop(kp, img_shape, crop=56):
    h, w = img_shape
    left, top = (w - crop) // 2, (h - crop) // 2
    return kp - np.array([left, top], dtype=np.float32), (crop, crop)


def flip_kp(kp, score, img_w):
    """Flip._flip_kps: mirror x on nonzero coords, swap left/right joints."""
    kp = kp.copy()
    score = score.copy()
    kp_x = kp[..., 0]
    kp_x[kp_x != 0] = img_w - kp_x[kp_x != 0]
    order = list(range(kp.shape[2]))
    for l, r in zip(LEFT_KP, RIGHT_KP):
        order[l], order[r] = r, l
    return kp[:, :, order], score[:, :, order]


def generate_pose_target(kp, score, img_shape, sigma=0.6, eps=1e-4):
    """GeneratePoseTarget with_kp, use_score. kp (M,T,V,2), returns (T,H,W,V)."""
    img_h, img_w = img_shape
    n_persons, n_frames, n_joints = kp.shape[:3]
    out = np.zeros((n_frames, img_h, img_w, n_joints), dtype=np.float32)
    for t in range(n_frames):
        for j in range(n_joints):
            heatmap = out[t, :, :, j]
            for m in range(n_persons):
                mu_x, mu_y = kp[m, t, j]
                max_value = score[m, t, j]
                if max_value < eps:
                    continue
                st_x = max(int(mu_x - 3 * sigma), 0)
                ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
                st_y = max(int(mu_y - 3 * sigma), 0)
                ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
                x = np.arange(st_x, ed_x, 1, np.float32)
                y = np.arange(st_y, ed_y, 1, np.float32)
                if not (len(x) and len(y)):
                    continue
                y = y[:, None]
                patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
                np.maximum(heatmap[st_y:ed_y, st_x:ed_x], patch * max_value,
                           out=heatmap[st_y:ed_y, st_x:ed_x])
    return out


def preprocess_window(kp, score, img_shape, clip_len=48, num_clips=10):
    """Full test pipeline for one (M,T,V,2) window. Returns (2*num_clips, V, clip_len, 56, 56)."""
    total = kp.shape[1]
    inds = uniform_sample_frames_test(total, clip_len, num_clips)
    kp = kp[:, inds].astype(np.float32)
    score = score[:, inds].astype(np.float32)

    kp, shape = pose_compact(kp, img_shape)
    kp, shape = resize_short_edge(kp, shape)
    kp, shape = center_crop(kp, shape)

    kp_f, score_f = flip_kp(kp, score, shape[1])
    imgs = np.concatenate([
        generate_pose_target(kp, score, shape),
        generate_pose_target(kp_f, score_f, shape),
    ])  # (2*num_clips*clip_len, 56, 56, V)

    imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
    imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
    return imgs.reshape((-1,) + imgs.shape[2:])


@torch.no_grad()
def score_window(model, device, kp, score, img_shape):
    """SMM probability for one window, mmaction average_clips='prob'."""
    imgs = preprocess_window(kp, score, img_shape)
    x = torch.from_numpy(imgs).to(device)
    logits = model(x)
    return F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
