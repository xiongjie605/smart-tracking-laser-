"""Web 控制台后端。

复用 backend/drivers/ai.py 与 backend/drivers/gimbal.py，
再加 Flask + MJPEG 推流，把"摄像头 + BPU 检测 + 云台"暴露到网页。

路由：
  GET  /                  frontend/index.html
  GET  /video_feed        MJPEG (multipart/x-mixed-replace)
  GET  /snap.jpg          单帧 jpg
  GET  /api/status        状态 JSON
  POST /api/tracking/start|stop
  POST /api/gimbal/manual  body {yaw,pitch,duration}
  POST /api/gimbal/stop
  GET  /api/recordings    录像列表
  GET  /recordings/<name> 录像文件
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from backend.utils.image import bgr_to_nv12, letterbox  # noqa: E402
from backend.drivers.ai import COCO, Detector  # noqa: E402

LOG_DIR = BASE / "logs"

# ---- config ------------------------------------------------------------


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


HTTP_PORT = _env_int("WEB_PORT", 8080)
CAMERA_INDEX = _env_int("CAMERA_INDEX", 0)
FRAME_W = _env_int("FRAME_W", 640)
FRAME_H = _env_int("FRAME_H", 480)
GIMBAL_PORT = os.environ.get("GIMBAL_PORT", "/dev/ttyUSB0")
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/opt/hobot/model/x5/basic/yolov5s_v6_640x640_nv12.bin",
)
CONF_THRESH = float(os.environ.get("CONF_THRESH", 0.4))
DISABLE_GIMBAL = os.environ.get("DISABLE_GIMBAL") == "1"
DISABLE_CAMERA = os.environ.get("DISABLE_CAMERA") == "1"
DISABLE_AI = os.environ.get("DISABLE_AI") == "1"


# ---- shared state -------------------------------------------------------


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None  # BGR ndarray
        self.latest_jpeg = None  # bytes
        self.latest_boxes = []  # [(x1,y1,x2,y2,conf,cls) ...]
        self.fps = 0.0
        self.det_count = 0
        self.tracking_enabled = False
        self.manual_gimbal = (0.0, 0.0)
        self.manual_gimbal_until = 0.0
        self.camera_ok = False
        self.gimbal_ok = False
        self.ai_ok = False
        self.last_error = ""


S = State()
detector = None
gimbal = None


# ---- helpers ------------------------------------------------------------


def _encode_jpeg(bgr, quality=80):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def _overlay(bgr, boxes, fps, tracking, gim_ok, det):
    draw = bgr
    for x1, y1, x2, y2, c, cls in boxes:
        cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{COCO[cls]} {c:.2f}"
        cv2.putText(draw, label, (int(x1), max(15, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(draw, (FRAME_W // 2, FRAME_H // 2), 6, (0, 255, 255), -1)
    txt = f"fps={fps:.1f}  det={'Y' if det else '.'}  track={'ON' if tracking else 'OFF'}  gim={'Y' if gim_ok else 'N'}"
    cv2.putText(draw, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    return draw


def _load_backends():
    global detector, gimbal
    if not DISABLE_AI:
        try:
            detector = Detector(MODEL_PATH, target_classes={0}, conf_thresh=CONF_THRESH)
            S.ai_ok = True
            print(f"[ai] model loaded: {MODEL_PATH}", flush=True)
        except Exception as e:
            S.ai_ok = False
            S.last_error = f"ai init fail: {e}"
            print(f"[ai] FAIL: {e}", flush=True)

    if not DISABLE_GIMBAL:
        try:
            from backend.drivers.gimbal import Gimbal  # type: ignore
            gimbal = Gimbal(GIMBAL_PORT)
            S.gimbal_ok = True
            print(f"[gim] serial open: {GIMBAL_PORT}", flush=True)
        except Exception as e:
            S.gimbal_ok = False
            print(f"[gim] FAIL (no gimbal? ok, fall back): {e}", flush=True)


# ---- capture loop -------------------------------------------------------


def _capture_loop():
    cap = None
    if not DISABLE_CAMERA:
        try:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 25)
            if not cap.isOpened():
                raise RuntimeError("VideoCapture not opened")
            S.camera_ok = True
            print(f"[cam] /dev/video{CAMERA_INDEX} opened {FRAME_W}x{FRAME_H} MJPG",
                  flush=True)
        except Exception as e:
            S.camera_ok = False
            S.last_error = f"camera fail: {e}"
            print(f"[cam] FAIL: {e}", flush=True)

    if not S.camera_ok:
        # 占位图：黑底白字
        ph = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        cv2.putText(ph, "NO CAMERA", (FRAME_W // 2 - 100, FRAME_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        while True:
            j = _encode_jpeg(ph)
            with S.lock:
                S.latest_frame = ph
                S.latest_jpeg = j
            time.sleep(0.5)

    fps_t, fps_n = time.time(), 0
    target_x, target_y = FRAME_W // 2, FRAME_H // 2
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        boxes = []
        if detector is not None:
            try:
                lb, r, dx, dy = letterbox(frame, 640)
                nv12 = bgr_to_nv12(lb)
                raw = detector.detect(nv12)
                for x1, y1, x2, y2, c, cls in raw:
                    boxes.append(((x1 - dx) / r, (y1 - dy) / r,
                                  (x2 - dx) / r, (y2 - dy) / r, c, cls))
            except Exception as e:
                print("[ai] forward fail:", e, flush=True)

        best = None
        best_area = 0
        for b in boxes:
            a = (b[2] - b[0]) * (b[3] - b[1])
            if a > 2500 and a > best_area:
                best_area, best = a, b

        now = time.time()
        if gimbal is not None and now < S.manual_gimbal_until:
            yaw, pitch = S.manual_gimbal
            gimbal.send(yaw, pitch)
        elif (gimbal is not None and S.tracking_enabled
              and best is not None):
            cx = (best[0] + best[2]) / 2
            cy = (best[1] + best[3]) / 2
            ex = target_x - cx
            ey = target_y - cy
            gimbal.track(ex, ey, target_x, target_y)

        n_det = 1 if best is not None else 0
        with S.lock:
            tracking_now = S.tracking_enabled
            gim_ok_now = S.gimbal_ok
            fps_now = S.fps

        draw = _overlay(frame, boxes, fps_now, tracking_now, gim_ok_now, n_det)

        j = _encode_jpeg(draw)
        with S.lock:
            S.latest_frame = draw
            S.latest_jpeg = j
            S.latest_boxes = boxes
            S.det_count = n_det

        fps_n += 1
        if time.time() - fps_t >= 2.0:
            now_fps = fps_n / (time.time() - fps_t)
            with S.lock:
                S.fps = now_fps
            fps_t = time.time()
            fps_n = 0


# ---- Flask app ----------------------------------------------------------


app = Flask(__name__, static_folder=None)
HTML_PATH = BASE / "frontend" / "index.html"


@app.route("/")
def index():
    if HTML_PATH.exists():
        return HTML_PATH.read_text(encoding="utf-8"), 200, {
            "Content-Type": "text/html; charset=utf-8"
        }
    return "<h1>frontend/index.html missing</h1>", 500


def _mjpeg_gen():
    while True:
        with S.lock:
            j = S.latest_jpeg
        if j is None:
            time.sleep(0.1)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(j)).encode() + b"\r\n\r\n" +
               j + b"\r\n")
        time.sleep(1 / 25)


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snap.jpg")
def snap():
    with S.lock:
        j = S.latest_jpeg
    if j is None:
        abort(503)
    return Response(j, mimetype="image/jpeg")


@app.route("/api/status")
def api_status():
    with S.lock:
        return jsonify(
            camera=S.camera_ok,
            ai=S.ai_ok,
            gimbal=S.gimbal_ok,
            tracking=S.tracking_enabled,
            fps=round(S.fps, 1),
            det=bool(S.det_count),
            manual_until=round(S.manual_gimbal_until, 2),
            now=time.time(),
            error=S.last_error,
        )


@app.route("/api/tracking/start", methods=["POST"])
def api_tracking_start():
    with S.lock:
        S.tracking_enabled = True
    return jsonify(ok=True)


@app.route("/api/tracking/stop", methods=["POST"])
def api_tracking_stop():
    with S.lock:
        S.tracking_enabled = False
    if gimbal is not None:
        gimbal.stop()
    return jsonify(ok=True)


@app.route("/api/gimbal/manual", methods=["POST"])
def api_gimbal_manual():
    if gimbal is None:
        return jsonify(ok=False, error="gimbal not available"), 503
    body = request.get_json(force=True, silent=True) or {}
    try:
        yaw = float(body.get("yaw", 0))
        pitch = float(body.get("pitch", 0))
    except Exception:
        return jsonify(ok=False, error="bad payload"), 400
    yaw = max(-26.0, min(26.0, yaw))
    pitch = max(-26.0, min(26.0, pitch))
    duration = float(body.get("duration", 0.3))
    duration = max(0.05, min(2.0, duration))
    with S.lock:
        S.manual_gimbal = (yaw, pitch)
        S.manual_gimbal_until = time.time() + duration
    return jsonify(ok=True, yaw=yaw, pitch=pitch,
                   until=round(S.manual_gimbal_until, 2))


@app.route("/api/gimbal/stop", methods=["POST"])
def api_gimbal_stop():
    if gimbal is not None:
        gimbal.stop()
    with S.lock:
        S.manual_gimbal_until = 0.0
    return jsonify(ok=True)


@app.route("/api/recordings")
def api_recordings():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(LOG_DIR.glob("track_*.avi"),
                    key=lambda x: -x.stat().st_mtime)[:20]:
        st = p.stat()
        items.append({"name": p.name, "size": st.st_size,
                      "mtime": int(st.st_mtime),
                      "url": f"/recordings/{p.name}"})
    return jsonify(items)


@app.route("/recordings/<path:name>")
def recordings_file(name):
    return send_from_directory(str(LOG_DIR), name, as_attachment=False)


# ---- main ---------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=HTTP_PORT)
    ap.add_argument("--no-gimbal", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-ai", action="store_true")
    args = ap.parse_args()
    if args.no_gimbal:
        os.environ["DISABLE_GIMBAL"] = "1"
    if args.no_camera:
        os.environ["DISABLE_CAMERA"] = "1"
    if args.no_ai:
        os.environ["DISABLE_AI"] = "1"

    print(f"[boot] {args.host}:{args.port}", flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _load_backends()
    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()
    app.run(host=args.host, port=args.port, threaded=True,
            debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
