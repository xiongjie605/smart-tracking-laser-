"""BPU 端 YOLOv5s 检测后处理。

模型: /opt/hobot/model/x5/basic/yolov5s_v6_640x640_nv12.bin
类别: COCO 80 类
3 路输出 (1, h, w, 3, 85) 各对应一个 anchor scale。
本文件实现:
  - decode_v5: 把 3 路 feature map 解码成 (x1,y1,x2,y2, conf, cls_id)
  - nms: 贪心 NMS
  - Detector: 整体封装, 提供 detect(frame) 接口
"""
import numpy as np
from hobot_dnn import pyeasy_dnn as pydnn


COCO = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)

ANCHORS = np.array([
    [[10, 13], [16, 30], [33, 23]],         # P3 stride 8
    [[30, 61], [62, 45], [59, 119]],        # P4 stride 16
    [[116, 90], [156, 198], [373, 326]],    # P5 stride 32
], dtype=np.float32)
STRIDES = (8, 16, 32)
NUM_CLASSES = 80
NUM_ANCHORS = 3


def _sigmoid(x):
    # 数值稳定版本
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def decode_v5(outs, conf_thresh=0.4, target_classes=None):
    """解码 3 路 YOLOv5 输出为 boxes list (x1,y1,x2,y2,conf,cls_id) at 640x640 scale."""
    if target_classes is None:
        target_classes = {0}
    target_mask = np.zeros(NUM_CLASSES, dtype=bool)
    for c in target_classes:
        target_mask[c] = True

    all_boxes = []
    for scale_idx, out in enumerate(outs):
        arr = np.array(out.buffer, copy=False).reshape(out.properties.shape)
        # arr shape: (1, h, w, 255) = (1, h, w, 3, 85)
        h, w = arr.shape[1], arr.shape[2]
        stride = STRIDES[scale_idx]
        anchors = ANCHORS[scale_idx]

        arr = arr.reshape(1, h, w, NUM_ANCHORS, 5 + NUM_CLASSES)
        t_xy = arr[..., 0:2]
        t_wh = arr[..., 2:4]
        t_obj = arr[..., 4:5]
        t_cls = arr[..., 5:]

        gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # YOLOv5 解码公式
        bx = (_sigmoid(t_xy[..., 0]) * 2 - 0.5 + gx[None, ..., None]) * stride
        by = (_sigmoid(t_xy[..., 1]) * 2 - 0.5 + gy[None, ..., None]) * stride
        bw = (_sigmoid(t_wh[..., 0]) * 2) ** 2 * anchors[None, None, None, :, 0]
        bh = (_sigmoid(t_wh[..., 1]) * 2) ** 2 * anchors[None, None, None, :, 1]

        x1 = bx - bw / 2
        y1 = by - bh / 2
        x2 = bx + bw / 2
        y2 = by + bh / 2

        obj = _sigmoid(t_obj)
        cls_prob = _sigmoid(t_cls)
        scores = obj * cls_prob  # (1, h, w, 3, 80)
        cls_max = scores.max(axis=-1)[0]  # (h, w, 3)
        cls_id = scores.argmax(axis=-1)[0]

        target_match = target_mask[cls_id]
        keep = (cls_max > conf_thresh) & target_match

        ys, xs, ai = np.where(keep)
        if len(ys) == 0:
            continue
        all_boxes.extend(zip(
            x1[0, ys, xs, ai].tolist(),
            y1[0, ys, xs, ai].tolist(),
            x2[0, ys, xs, ai].tolist(),
            y2[0, ys, xs, ai].tolist(),
            cls_max[ys, xs, ai].tolist(),
            cls_id[ys, xs, ai].astype(int).tolist()
        ))
    return all_boxes


def nms(boxes, iou_thresh=0.45):
    """类内 NMS, 按 conf 降序贪心。输入 boxes = [(x1,y1,x2,y2,c,cls), ...]"""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        rest = []
        bx1, by1, bx2, by2 = best[0], best[1], best[2], best[3]
        b_area = (bx2 - bx1) * (by2 - by1)
        for b in boxes:
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            ix1, iy1 = max(x1, bx1), max(y1, by1)
            ix2, iy2 = min(x2, bx2), min(y2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            area = (x2 - x1) * (y2 - y1)
            union = area + b_area - inter
            iou = inter / max(union, 1e-9)
            if iou < iou_thresh:
                rest.append(b)
        boxes = rest
    return keep


class Detector:
    """BPU YOLOv5s 实时检测器 (单类 person)."""

    def __init__(self, model_path: str, target_classes=None,
                 conf_thresh=0.4, iou_thresh=0.45, imgsz=640):
        self.model = pydnn.load(model_path)[0]
        self.target_classes = target_classes or {0}
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz

    def detect(self, nv12: np.ndarray):
        """输入: NV12 bytes (614400). 输出: boxes (x1,y1,x2,y2,conf,cls) at 640x640 scale."""
        outs = self.model.forward([nv12])
        boxes = decode_v5(outs, self.conf_thresh, self.target_classes)
        return nms(boxes, self.iou_thresh)
