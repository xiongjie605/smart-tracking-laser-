"""图像转换: BGR -> NV12 (供 BPU 吃) + letterbox。"""
import cv2
import numpy as np


def letterbox(img: np.ndarray, new: int = 640, color=(114, 114, 114)):
    """保持比例缩放到 new×new, 灰条填充。返回 (canvas, ratio, dx, dy)."""
    h, w = img.shape[:2]
    r = new / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    img2 = cv2.resize(img, (nw, nh))
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    dx, dy = (new - nw) // 2, (new - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = img2
    return canvas, r, dx, dy


def bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    """BGR -> NV12 (YUV 4:2:0 semi-planar).

    cv2 COLOR_BGR2YUV_I420 把 chroma 存成 (h/4, w) 的"宽"格式,
    每两列相邻同值 (1/2 横向子采样)。这里转成标准 NV12 (h/2, w) 交错 UV。
    """
    h, w = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    Y = yuv[:h].flatten()
    U_raw = yuv[h:h + h // 4]   # cv2 存储 (h/4, w)
    V_raw = yuv[h + h // 4:]
    U_ch = U_raw[:, ::2]          # 1/2 横向子采样后 (h/4, w/2)
    V_ch = V_raw[:, ::2]
    U_v = np.repeat(U_ch, 2, axis=0)   # 纵向 2x → (h/2, w/2)
    V_v = np.repeat(V_ch, 2, axis=0)
    UV = np.empty((h // 2, w), dtype=np.uint8)
    UV[:, 0::2] = U_v   # NV12: U 在偶数列
    UV[:, 1::2] = V_v
    return np.concatenate([Y, UV.flatten()])
