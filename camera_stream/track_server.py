#!/usr/bin/env python3
import sys, os, time, json, subprocess, threading, collections, signal, io, re, math
import cv2, numpy as np
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, "/root/web_control")
from backend.drivers.ai import Detector
from backend.drivers.gimbal import Gimbal
from backend.utils.image import letterbox, bgr_to_nv12
try:
    import torch
    from bytetracker import BYTETracker
    _HAS_BYTETRACK = True
except Exception as _e:
    _HAS_BYTETRACK = False
    print(f"[track] bytetracker NOT available: {_e}")

COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

PORT = 8080; CAM_ID = 0; CAM_W, CAM_H = 640, 360; JPEG_QUALITY = 50
N_DETECT = 4; KP = 0.035; DEAD_ZONE = 4          # 1 -> 4: BPU 噪声缓冲
MAX_SPEED = 26.0
TARGET_X, TARGET_Y = 320, 240; MANUAL_TIMEOUT_S = 30.0; MANUAL_AIM_SPEED = 8.0
RECORD_DIR = "/userdata/recordings"; RECORD_FPS = 25.0
VOICE_DIR = "/userdata/voice"; VOICE_ALSA = "plughw:0,0"; SPK_ALSA = "plughw:0,0"
ALSA_CARD = "0"; ALSA_CTRL = "PCM"
PID_SNAP = 0.5          # 输出 < 0.5 rpm -> 0 (防小幅晃动)
RPM_RATE_LIMIT = 2.0    # 每秒最多变化 2 rpm
EMA_ALPHA = 0.30        # bbox 中心 EMA 平滑（默认 fallback）
EMA_ALPHA_FAST = 0.55     # TRACKING 状态: 快速跟随（解决目标抖动后很久才追到位）
EMA_ALPHA_HOLD = 0.18     # HOLD 状态: 强平滑防抖
KP = 0.035
KP_FAST = 0.060           # |误差|>25px 时: 快速收敛
UVC_DEV = "/dev/video0"
UVC_CTRLS = {"zoom_absolute":(0,260),"focus_absolute":(0,160),"focus_automatic_continuous":(0,1),
             "pan_absolute":(-648000,648000),"tilt_absolute":(-648000,648000)}
os.makedirs(RECORD_DIR, exist_ok=True)

_lock = threading.Lock()
_frame_lock = threading.Lock()  # 保护 _latest_frame (raw BGR ndarray, 追踪线程写, BPU/render 线程读)
_latest_frame = None            # raw BGR ndarray, 追踪主线程写, render_loop/bpu_loop 读
_bboxes_lock = threading.Lock()  # 保护 _latest_bboxes 和 _detect_busy
_detect_pending_n = 0            # 待 detect 的 _n_frame (主循环写, detect 读)
_detect_busy = False             # BPU detect 线程是否在跑
_detect_event = threading.Event()  # 通知 detect 线程有新任务
_last_bpu = -1                   # 上次 BPU 完成时的 _n_frame (模块级, detect 写, 主循环读)
_tracking = True; _target_cls = "person"
_manual_mode = False; _manual_last_time = 0.0
_manual_no_timeout = False  # True=智能家居模式, 手动永不超时
_cmd_ring = collections.deque(maxlen=200); _n_cmd_total = 0
_recording = False; _v4l2_recording = False; _ffmpeg_proc = None
_record_path = ""; _record_start_ts = 0.0; _record_n_frames = 0; _record_finished_at = 0.0
_ffmpeg_frame_buf = b""
_cap = None; _det = None; _gim = None
_laser_drawing = False   # 激光绘制锁, 防并发
_latest_jpeg = None; _latest_bboxes = []
_capture_actual_fps = 0.0; _bpu_last_fps = 0.0
_n_frame = 0; _ever_tracked = False
_audio_cap = None; _audio_clients = []; _audio_lock = threading.Lock(); _audio_sample_rate = 16000
_ptt_proc = None; _ptt_data = bytearray(); _ptt_lock = threading.Lock()
_ptt_last_file = ""
_ptt_play_proc = None  # 当前 aplay 自动播放进程，_ptt_start 录音前会清理

# 多目标追踪状态 (改用 ByteTrack Kalman 滤波 + 速度模型, 解决过冲)
_byte_tracker = None   # BYTETracker 实例, main() 初始化  # [{'id':N,'cx':f,'cy':f,'conf':f,'cls':int,'bbox':(x1,y1,x2,y2),'lost_frames':int,'age':int}]
_next_target_id = 0
_primary_id = -1  # 当前锁定追踪的目标 ID，-1 表示未锁定
_PRIMARY_CONF_MARGIN = 0.15  # 切换锁定目标所需的最低置信度超出量
TRACK_POSITION_THRESHOLD = 120.0  # 两帧之间目标位置最大像素差，超过此值视为新目标或误检
TRACK_LOST_MAX = 5  # 连续丢失 N 帧后移除该目标
MAX_TARGETS = 10  # 同时追踪的最大目标数
TRACK_CLASSES = {"person", "cat", "dog"}  # 只追踪这些类别, 忽略其他

# 追踪状态机 (来自 tracktest, 迟滞死区 + EMA 平滑)
STATE_IDLE = 0; STATE_TRACKING = 1; STATE_HOLDING = 2
_track_state = STATE_IDLE
_hold_counter = 0
_ema_cx, _ema_cy = 0.0, 0.0
_ema_initialized = False
_last_yaw, _last_pitch = 0.0, 0.0
_prev_ex, _prev_ey = 0.0, 0.0   # 用于 HOLD 速度检测 (velocity-based release)
DEAD_ZONE_ENTER = 14  # 进入 HOLD 的阈值 (像素, 原 18 → 14: 早点停, 减少抖动)
DEAD_ZONE_EXIT = 22   # 退出 HOLD 的阈值 (像素, 原 40 → 22: 大幅降低, 避免卡死)
HOLD_MIN_FRAMES = 3   # 进入 HOLD 后至少保持 N 帧 (原 8 → 3: 快速响应),'cx':f,'cy':f,'conf':f,'cls':int,'bbox':(x1,y1,x2,y2),'lost_frames':int,'age':int}]

def _uvc_get(ctrl):
    try: return subprocess.check_output(["v4l2-ctl","-d",UVC_DEV,"--get-ctrl="+ctrl],text=True,timeout=3).strip()
    except: return ""

def _uvc_set(ctrl, val):
    try: subprocess.run(["v4l2-ctl","-d",UVC_DEV,"--set-ctrl="+ctrl+"="+str(val)],capture_output=True,timeout=3); return True
    except: return False

def _async_aim(yaw_deg, pitch_deg):
    """手动旋转 yaw_deg/pitch_deg 度. 设 _manual_mode, 30s 超时回自动."""
    set_manual_mode()
    def _do():
        if _gim is None: return
        # 用 MANUAL_AIM_SPEED=8 rpm 转 duration 秒
        # 角度速: 8 rpm = 48 deg/s (8*360/60=48)
        d = max(abs(yaw_deg), abs(pitch_deg)) / (MANUAL_AIM_SPEED * 360.0 / 60.0)
        yaw_rpm = MANUAL_AIM_SPEED if yaw_deg > 0 else -MANUAL_AIM_SPEED if yaw_deg < 0 else 0
        pitch_rpm = MANUAL_AIM_SPEED if pitch_deg > 0 else -MANUAL_AIM_SPEED if pitch_deg < 0 else 0
        _gim.send(yaw_rpm, pitch_rpm); time.sleep(d); _gim.send(0,0)
    threading.Thread(target=_do, daemon=True).start()

def _ptt_start():
    """启动按住说话录音 (独占 USB 麦)."""
    global _ptt_proc, _ptt_data, _audio_cap, _ptt_play_proc
    with _ptt_lock:
        if _ptt_proc is not None:
            return "busy"
        # 先杀掉上次自动播放的 aplay，避免与新 arecord 抢 plughw:0,0
        if _ptt_play_proc is not None:
            try:
                if _ptt_play_proc.poll() is None:
                    _ptt_play_proc.terminate()
                    try: _ptt_play_proc.wait(timeout=1.0)
                    except Exception:
                        try: _ptt_play_proc.kill() if _ptt_play_proc else None
                        except: pass
            except: pass
            _ptt_play_proc = None
            # ALSA pcm 子设备可能还没释放, 等设备真正空闲（最多 2s）
            for _wait in range(20):
                if subprocess.run(["fuser", "-s", "/dev/snd/pcmC0D0p"],
                                  capture_output=True).returncode != 0:
                    break
                time.sleep(0.1)
        with _audio_lock:
            if _audio_cap is not None:
                try: _audio_cap.terminate(); _audio_cap.wait(timeout=2)
                except: pass
                _audio_cap = None
        _ptt_data = bytearray()
        try:
            _ptt_proc = subprocess.Popen(
                ["arecord", "-D", VOICE_ALSA, "-f", "S16_LE",
                 "-r", str(_audio_sample_rate), "-c", "1", "-t", "raw"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            threading.Thread(target=_ptt_collect, daemon=True).start()
            print(f"[ptt] start pid={_ptt_proc.pid}")
            return "ok"
        except Exception as e:
            print(f"[ptt] FAIL: {e}")
            _ptt_proc = None
            return f"err={e}"

def _ptt_collect():
    global _ptt_proc
    while _ptt_proc is not None and _ptt_proc.poll() is None:
        try:
            chunk = _ptt_proc.stdout.read(4096)
            if not chunk: break
            with _ptt_lock:
                if _ptt_proc is not None:
                    _ptt_data.extend(chunk)
        except Exception:
            break

def _alsa_get_volume():
    """读 USB 扬声器 PCM 音量 (0-100%), 失败返回 None."""
    try:
        out = subprocess.check_output(
            ["amixer", "-c", ALSA_CARD, "get", ALSA_CTRL], text=True, timeout=3)
        import re as _re
        m = _re.search(r"\[(\d+)%\]", out)
        if m: return int(m.group(1))
    except Exception as e:
        print(f"[alsa] get volume FAIL: {e}")
    return None

def _alsa_set_volume(pct):
    """设 USB 扬声器 PCM 音量 (0-100%)."""
    pct = max(0, min(100, pct))
    try:
        r = subprocess.run(
            ["amixer", "-c", ALSA_CARD, "set", ALSA_CTRL, f"{pct}%"],
            capture_output=True, text=True, timeout=3)
        print(f"[alsa] set PCM={pct}% -> rc={r.returncode}")
        return r.returncode == 0
    except Exception as e:
        print(f"[alsa] set volume FAIL: {e}")
        return False

def _ptt_stop():
    """停止 PTT 录音, 保存到 /userdata/voice/, 返回 WAV 字节 (含 wav 头)."""
    global _ptt_proc, _ptt_data, _ptt_last_file, _ptt_play_proc
    with _ptt_lock:
        if _ptt_proc is None:
            return b""
        proc = _ptt_proc
        _ptt_proc = None
        data = bytes(_ptt_data)
        _ptt_data = bytearray()
    try: proc.terminate(); proc.wait(timeout=2)
    except: proc.kill() if proc else None
    import wave, io, os
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(_audio_sample_rate)
        w.writeframes(data)
    wav_bytes = buf.getvalue()
    # 保存到 /userdata/voice/ptt_<时间戳>.wav (供后续 /voice/play/<filename> 播放)
    try:
        os.makedirs(VOICE_DIR, exist_ok=True)
        fname = f"ptt_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        fpath = os.path.join(VOICE_DIR, fname)
        with open(fpath, "wb") as f:
            f.write(wav_bytes)
        _ptt_last_file = fname
        print(f"[ptt] saved {fpath} ({len(wav_bytes)} bytes)")
    except Exception as e:
        print(f"[ptt] save FAIL: {e}")
        _ptt_last_file = ""
    # 松开后自动通过外部扬声器播放 (后台, 不阻塞响应)，保存引用以便下次 PTT 录音前清理
    try:
        _ptt_play_proc = subprocess.Popen(["aplay", "-D", SPK_ALSA, "-q", fpath],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[ptt] auto-play {fpath} pid={_ptt_play_proc.pid}")
    except Exception as e:
        print(f"[ptt] auto-play FAIL: {e}")
        _ptt_play_proc = None
    print(f"[ptt] stop {len(data)} bytes raw -> {len(wav_bytes)} bytes wav")
    return wav_bytes

def _stt_wav(wav_bytes):
    """对 WAV 字节做 STT, 返回识别文字."""
    if not wav_bytes or len(wav_bytes) < 100:
        return ""
    try:
        import wave, io, numpy as np, sherpa_onnx
        model_dir = "/opt/sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            audio = w.readframes(n)
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=f"{model_dir}/model.int8.onnx",
            tokens=f"{model_dir}/tokens.txt",
            num_threads=2, debug=False,
        )
        stream = rec.create_stream()
        stream.accept_waveform(sample_rate=sr, waveform=samples)
        rec.decode_stream(stream)
        text = stream.result.text.strip()
        print(f"[stt] {len(wav_bytes)} bytes -> '{text}'")
        return text
    except Exception as e:
        print(f"[stt] FAIL: {e}")
        return f"[err: {e}]"

def _start_audio_capture():
    """启动 USB 麦后台采集 (16kHz mono PCM), 只在第一次调用时启动."""
    global _audio_cap
    with _audio_lock:
        if _audio_cap is not None:
            return True
        # 启动前确保无残留 arecord + aplay (避免 ALSA device busy)
        subprocess.run(["pkill", "-9", "-f", "arecord.*" + VOICE_ALSA.replace(",", "\\,")],
                       capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "aplay.*" + SPK_ALSA.replace(",", "\\,")],
                       capture_output=True)
        time.sleep(0.5)
        for attempt in range(3):
            try:
                _audio_cap = subprocess.Popen(
                    ["arecord", "-D", VOICE_ALSA, "-f", "S16_LE",
                     "-r", str(_audio_sample_rate), "-c", "1", "-t", "raw"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
                time.sleep(0.6)  # 等 arecord 真正打开设备
                if _audio_cap.poll() is not None:
                    print(f"[audio] attempt {attempt+1}: died rc={_audio_cap.returncode}")
                    _audio_cap = None
                    time.sleep(0.4)
                    continue
                threading.Thread(target=_broadcast_audio, daemon=True).start()
                print(f"[audio] capture started pid={_audio_cap.pid} attempt={attempt+1}")
                return True
            except Exception as e:
                print(f"[audio] FAIL attempt {attempt+1}: {e}")
                _audio_cap = None
                time.sleep(0.4)
        return False

def _broadcast_audio():
    """从 arecord stdout 读 PCM 块, 软降噪后广播给所有客户端 (HTTP chunked)."""
    global _audio_cap
    import numpy as _np
    from scipy.signal import butter as _butter, sosfilt as _sosfilt
    NOISE_GATE = 5     # RMS 低于此值视为噪音, 输出静音 (灵敏度)
    GAIN = 0.4         # 软降增益 (麦 95% 太大)
    TARGET_RMS = 4000  # 自动增益目标 RMS (适中, 避免过度放大环境噪音)
    MAX_GAIN = 3.0     # AGC 上限 3x (4 阶 Butterworth LPF 在 3919Hz 只衰减 -26dB, 8x 会被 AGC 抵消)
    # 频谱诊断发现 USB 麦有 3919Hz 强尖峰 (99dB) → 尖锐刺耳声
    # 用 Butterworth 滤波器: HPF 2 阶 fc=150Hz + LPF 4 阶 fc=2200Hz (综合 -80dB/dec 滚降)
    # 3919Hz 处合并衰减 ~-26dB, 用户语音 300-2000Hz 几乎不衰减
    from scipy.signal import iirnotch as _iirnotch, lfilter as _lfilter
    _SOS_HPF = _butter(2, 150, btype='high', fs=16000, output='sos')
    _SOS_LPF = _butter(4, 2200, btype='low', fs=16000, output='sos')
    # Notch @ 3919Hz Q=30 (带宽 ±65Hz) 精准滤掉 USB 麦时钟泄漏的尖锐峰
    _NOTCH_B, _NOTCH_A = _iirnotch(w0=3919, Q=30, fs=16000)
    _smooth_scale = 1.0    # AGC scale 平滑 (attack/release one-pole), 消除 click 噪声
    _S_ATTACK = 0.15       # 上升快 (突然有声音)
    _S_RELEASE = 0.03      # 下降慢 (声音消失后慢慢回落)
    _notch_zi = None       # notch 滤波器跨 chunk 状态
    _diag_n = 0
    while _audio_cap is not None and _audio_cap.poll() is None:
        try:
            chunk = _audio_cap.stdout.read(4096)
            if not chunk:
                break
        except Exception:
            break
        # 软降噪: 噪声门 + 增益衰减
        try:
            samples = _np.frombuffer(chunk, dtype=_np.int16).astype(_np.float32)
            rms = float(_np.sqrt(_np.mean(samples * samples))) if len(samples) else 0
            _diag_n += 1
            if _diag_n in (1, 5, 20, 100, 500) or _diag_n % 500 == 0:
                mx = int(samples.max()) if len(samples) else 0
                mn = int(samples.min()) if len(samples) else 0
                nz = int((samples != 0).sum()) if len(samples) else 0
                print(f"[audio] #{_diag_n} rms={rms:.0f} max={mx} min={mn} nonzero={nz}/{len(samples)} clients={len(_audio_clients)}")
            if rms < NOISE_GATE:
                chunk = b"\x00" * len(chunk)  # 静音
            else:
                # DC removal (去直流偏置)
                samples = samples - samples.mean()
                # scipy.signal.sosfilt 是 vectorized C, 比一阶 IIR for 循环快 100x
                # 2 阶 Butterworth HPF fc=150Hz 衰减电机嗡 (~175Hz)
                samples = _sosfilt(_SOS_HPF, samples)
                # 4 阶 Butterworth LPF fc=2200Hz 衰减 3919Hz 尖峰 (-26dB)
                samples = _sosfilt(_SOS_LPF, samples)
                # Notch @ 3919Hz Q=30 精准滤除 USB 麦时钟泄漏尖峰 (-30dB)
                # lfilter 保留跨 chunk state 避免 transient
                if _notch_zi is None:
                    _notch_zi = _np.zeros(max(len(_NOTCH_A), len(_NOTCH_B)) - 1, dtype=_np.float32)
                samples, _notch_zi = _lfilter(_NOTCH_B, _NOTCH_A, samples, zi=_notch_zi)
                # AGC: 用滤波后 rms 算 scale (避免低频嗡 ~175/386Hz 驱动过度放大)
                post_rms = float(_np.sqrt(_np.mean(samples * samples))) if len(samples) else 0
                if post_rms < TARGET_RMS:
                    target_scale = min(TARGET_RMS / max(post_rms, 1.0), MAX_GAIN)
                    # AGC scale 平滑 (one-pole attack/release), 消除 chunk 边界 gain 跳变产生的 click
                    if target_scale > _smooth_scale:
                        _smooth_scale = _smooth_scale + (target_scale - _smooth_scale) * _S_ATTACK
                    else:
                        _smooth_scale = _smooth_scale + (target_scale - _smooth_scale) * _S_RELEASE
                    scale = _smooth_scale
                    samples = _np.clip(samples * scale, -32768, 32767).astype(_np.int16)
                else:
                    _smooth_scale = _smooth_scale + (1.0 - _smooth_scale) * _S_RELEASE  # 大声音时也慢慢回 1
                    samples = _np.clip(samples * GAIN, -32768, 32767).astype(_np.int16)
                chunk = samples.tobytes()
        except Exception:
            pass
        with _audio_lock:
            clients = list(_audio_clients)
        for wfile, lock in clients:
            try:
                with lock:
                    wfile.write(chunk)
                    wfile.flush()
            except Exception:
                with _audio_lock:
                    if (wfile, lock) in _audio_clients:
                        _audio_clients.remove((wfile, lock))
    print("[audio] capture ended")
    _audio_cap = None

def set_manual_mode(no_timeout=False):
    """no_timeout=True: 智能家居模式手动永不超时, 只有 /mode/auto 能切回自动."""
    global _manual_mode, _manual_last_time, _manual_no_timeout
    _manual_mode = True
    _manual_last_time = time.monotonic()
    _manual_no_timeout = no_timeout

def check_manual_timeout():
    global _manual_mode, _manual_last_time, _manual_no_timeout
    if _manual_no_timeout:
        return False  # 智能家居模式: 永不超时
    if _manual_mode and _manual_last_time > 0 and (time.monotonic() - _manual_last_time) > MANUAL_TIMEOUT_S:
        _manual_mode = False; _manual_last_time = 0; return True
    return False

def _record_cmd(yaw,pitch,state,frame,lost):
    global _n_cmd_total; _n_cmd_total += 1
    _cmd_ring.append({"yaw":round(yaw,3),"pitch":round(pitch,3),"state":state,"frame":frame,"lost":lost})

def _start_recording(auto_stop_s=0.0):
    global _recording, _ffmpeg_proc, _record_path, _record_start_ts, _record_n_frames, _v4l2_recording, _cap
    if _recording: return os.path.basename(_record_path) if _record_path else "busy"
    ts = time.strftime("%Y%m%d_%H%M%S"); fname = f"vid_{ts}.mp4"
    fpath = os.path.join(RECORD_DIR, fname)
    _v4l2_recording = True
    # 停止音频共享采集 (ffmpeg 需要独占 USB 麦)
    global _audio_cap
    with _audio_lock:
        if _audio_cap is not None:
            try: _audio_cap.terminate(); _audio_cap.wait(timeout=2)
            except: _audio_cap.kill() if _audio_cap else None
            _audio_cap = None
            print("[audio] capture stopped for recording")
    if _cap is not None: _cap.release(); _cap = None; time.sleep(0.3)
    cmd = ["ffmpeg","-y",
           "-f","v4l2","-input_format","mjpeg","-video_size","640x480","-framerate","30","-i","/dev/video0",
           "-f","alsa","-ac","1","-ar","16000","-i",VOICE_ALSA,
           "-c:v","libx264","-crf","18","-preset","fast","-pix_fmt","yuv420p",
           "-c:a","aac","-b:a","64k","-shortest",
           fpath]
    try:
        _ffmpeg_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e: print(f"[rec] FAIL start: {e}"); _ffmpeg_proc=None; _v4l2_recording=False; return ""
    _record_path=fpath; _record_start_ts=time.time(); _record_n_frames=0; _recording=True
    print(f"[rec] START -> {fpath} (30fps, auto_stop={auto_stop_s}s)")
    if auto_stop_s>0: threading.Thread(target=lambda: (time.sleep(auto_stop_s),_recording and _stop_recording()), daemon=True).start()
    return fname

def _stop_recording():
    global _recording, _ffmpeg_proc, _record_path, _record_n_frames, _v4l2_recording, _cap, _ffmpeg_frame_buf
    if not _recording: return ""
    _recording=False; proc=_ffmpeg_proc; _ffmpeg_proc=None
    if proc:
        try: proc.terminate(); proc.wait(timeout=3)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=1)
        except: pass
    _ffmpeg_frame_buf=b""
    for _ in range(3):
        try:
            if _cap is not None: _cap.release()
            _cap=cv2.VideoCapture(0,cv2.CAP_ANY); _cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*"MJPG"))
            _cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W); _cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
            _cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            if _cap.isOpened(): break
        except: pass
        time.sleep(0.5)
    _v4l2_recording=False
    _record_finished_at = time.time()  # 标记完成时间, 给前端验证
    print(f"[rec] STOP cap={_cap.isOpened() if _cap else False} finished_at={_record_finished_at:.1f}")
    fn=os.path.basename(_record_path) if _record_path else ""; sk=0
    if _record_path and os.path.exists(_record_path): sk=os.path.getsize(_record_path)/1024
    print(f"[rec] STOP {fn} dur={time.time()-_record_start_ts:.1f}s size={sk:.0f}KB")
    return f"file={fn}\n"

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/voice/stt":
            length = int(self.headers.get("Content-Length", "0"))
            wav = self.rfile.read(length) if length > 0 else b""
            text = _stt_wav(wav)
            self._send_json({"text": text})
            return
        self.send_error(404)

    def do_GET(self):
        qs={}; path=self.path
        global _tracking, _target_cls, _manual_mode
        if '?' in path: path,qs_str=path.split('?',1); qs={k:v for k,v in [x.split('=') for x in qs_str.split('&') if '=' in x]}
        def st(t,c='text/plain'): self._send_text(t.encode() if isinstance(t,str) else t)
        if path=="/": self._serve_panel()
        elif path=="/health": st("ok")
        elif path=="/stats": self._send_json({"capture_fps":_capture_actual_fps,"bpu_fps":_bpu_last_fps,"last_bboxes":len(_latest_bboxes),"tracking":_tracking})
        elif path=="/frame.jpg": self._serve_jpeg()
        elif path=="/video_feed": self._stream_mjpeg()
        elif path=="/track/on" or path=="/track/off": _tracking=path.endswith("/on"); st(f"tracking={_tracking}")
        elif path=="/target": _target_cls=qs.get("cls","person"); st(f"target={_target_cls}")
        elif path in("/ptz","/ptz/manual"):
            y=float(qs.get("yaw","0")); p=float(qs.get("pitch","0"))
            y=max(-26,min(26,y)); p=max(-26,min(26,p))
            if _gim: _gim.send(y,p); st(f"ptz yaw={y:.2f} pitch={p:.2f}\\n")
        elif path in("/aim","/manual/aim"):
            y=float(qs.get("yaw","0")); p=float(qs.get("pitch","0"))
            y=max(-30,min(30,y)); p=max(-30,min(30,p))
            _async_aim(y,p); st(f"aim yaw={y:+.1f}deg pitch={p:+.1f}deg\\n")
        elif path=="/mode/manual":
            set_manual_mode(no_timeout=True)  # 智能家居: 手动模式持久, 不自动回 auto
            st("manual mode, tracking disabled\n")
        elif path=="/mode/auto": _manual_mode=False; st("auto track mode resumed\\n")
        elif path=="/zoom":
            v=int(qs.get("value","0")); _uvc_set("zoom_absolute",max(0,min(260,v))); st(f"zoom={v}\\n")
        elif path=="/focus":
            v=int(qs.get("value","0")); _uvc_set("focus_absolute",max(0,min(160,v))); st(f"focus={v}\\n")
        elif path=="/focus/auto":
            on=int(qs.get("on","1")); _uvc_set("focus_automatic_continuous",1 if on else 0); st(f"focus_auto={on}\\n")
        elif path=="/uvc/get":
            c=qs.get("ctrl","zoom_absolute"); st(_uvc_get(c)+"\n")
        elif path.startswith("/recordings/"):
            fn=path[len("/recordings/"):]
            if ".." in fn or "/" in fn or not fn.endswith(".mp4"): self.send_error(400); return
            fp=os.path.join(RECORD_DIR,fn)
            if not os.path.isfile(fp): self.send_error(404); return
            with open(fp,"rb") as f: d=f.read()
            self.send_response(200); self.send_header("Content-Type","video/mp4"); self.send_header("Content-Length",str(len(d)))
            self.send_header("Content-Disposition",f'attachment; filename="{fn}"'); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers(); self.wfile.write(d)
        elif path=="/record/start": st(_start_recording(float(qs.get("auto","0")))+"\n")
        elif path=="/record/stop": st(_stop_recording()+"\n")
        elif path.startswith("/laser/pattern/"):
            name = path[len("/laser/pattern/"):].strip("/")
            ok, msg = _laser_draw_pattern(name)
            self._send_json({"status": "ok" if ok else "fail", "msg": msg, "pattern": name})
        elif path=="/laser/stop":
            # 紧急停止: 派一个零速指令即可 (_laser_drawing 自然会退出)
            try:
                if _gim: _gim.send(0, 0)
                self._send_json({"status":"ok"})
            except Exception as e:
                self.send_error(500, str(e))
        elif path=="/laser/status":
            self._send_json({"drawing": _laser_drawing, "gimbal": _gim is not None})
        elif path=="/record/status":
            # 前端轮询用: 返回当前是否在录制, 以及期望文件名是否已 finalize
            # 当 _recording=False 且 _record_path 存在, 表示上次录制完成
            self._send_json({
                "recording": _recording,
                "file": os.path.basename(_record_path) if _record_path else "",
                "size": os.path.getsize(_record_path) if _record_path and os.path.exists(_record_path) else 0,
                "elapsed": time.time() - _record_start_ts if _recording and _record_start_ts else 0,
            })
            return
        elif path=="/record/list":
            try:
                fs=sorted([f for f in os.listdir(RECORD_DIR) if f.endswith(".mp4")],key=lambda x:os.path.getmtime(os.path.join(RECORD_DIR,x)),reverse=True)[:20]
                st("\n".join(f"file={f} size={os.path.getsize(os.path.join(RECORD_DIR,f))/1024:.0f}KB time={time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(os.path.getmtime(os.path.join(RECORD_DIR,f))))}" for f in fs)+"\n")
            except: st("error\\n")
        elif path=="/voice/list":
            try:
                fs=sorted([f for f in os.listdir(VOICE_DIR) if f.endswith(".wav")])
                self._send_json({"voices":[{"name":f[:-4],"file":f} for f in fs]})
            except Exception as e:
                self._send_json({"voices":[],"error":str(e)})
        elif path=="/voice/ptt/latest":
            if _ptt_last_file:
                self._send_json({"file":_ptt_last_file, "url":"/voice/play/"+_ptt_last_file})
            else:
                self.send_error(404, "no ptt recording yet")
        elif path.startswith("/voice/play/"):
            fn=path[len("/voice/play/"):]
            if ".." in fn or "/" in fn or not fn.endswith(".wav"): self.send_error(400); return
            fp=os.path.join(VOICE_DIR,fn)
            if not os.path.isfile(fp): self.send_error(404); return
            try:
                subprocess.Popen(["aplay","-D",SPK_ALSA,"-q",fp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                st(f"play={fn}\n")
            except Exception as e:
                st(f"err={e}\n")
        elif path.startswith("/voice/wav/"):
            # 直接返回 WAV 字节 (前端 audio 元素用)
            fn=path[len("/voice/wav/"):]
            if ".." in fn or "/" in fn or not fn.endswith(".wav"): self.send_error(400); return
            fp=os.path.join(VOICE_DIR,fn)
            if not os.path.isfile(fp): self.send_error(404); return
            try:
                with open(fp,"rb") as f: d=f.read()
                self.send_response(200)
                self.send_header("Content-Type","audio/wav")
                self.send_header("Content-Length",str(len(d)))
                self.send_header("Access-Control-Allow-Origin","*")
                self.end_headers()
                self.wfile.write(d)
            except Exception as e:
                self.send_error(500, str(e))
        elif path=="/voice/ptt/start":
            st(_ptt_start()+"\n")
        elif path=="/voice/ptt/stop":
            wav = _ptt_stop()
            if not wav:
                self.send_error(400, "ptt not active"); return
            # 返回 JSON, 不返回 wav bytes (服务端已自动 aplay 播放到扬声器)
            self._send_json({"status": "ok", "bytes": len(wav), "played": True})
        elif path=="/voice/stt":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                wav = self.rfile.read(length) if length > 0 else b""
            except Exception:
                wav = b""
            text = _stt_wav(wav)
            self._send_json({"text": text})
        elif path=="/audio/volume":
            pct = _alsa_get_volume()
            self._send_json({"percent": pct if pct is not None else -1})
        elif path=="/audio/volume/set":
            try: pct = int(qs.get("value", "50"))
            except: pct = 50
            clamped = max(0, min(100, pct))
            ok = _alsa_set_volume(clamped)
            self._send_json({"percent": clamped if ok else -1, "ok": ok, "requested": pct})
        elif path=="/audio_stream":
            # 直接写裸 PCM (16kHz mono S16_LE), 不加 chunked 编码
            # HTTP/1.0 无 Content-Length, 客户端读到 socket 关闭为止
            if not _start_audio_capture():
                self.send_error(503, "audio capture failed"); return
            self.send_response(200)
            self.send_header("Content-Type", "audio/pcm")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Audio-Sample-Rate", str(_audio_sample_rate))
            self.send_header("X-Audio-Channels", "1")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            client_lock = threading.Lock()
            with _audio_lock:
                _audio_clients.append((self.wfile, client_lock))
            try:
                # 客户端断开检测: 写心跳, 写失败 = socket 关闭
                while True:
                    time.sleep(2)
                    try:
                        with client_lock:
                            self.wfile.write(b"")
                            self.wfile.flush()
                    except Exception:
                        break
                    with _audio_lock:
                        if (self.wfile, client_lock) not in _audio_clients:
                            break
            except Exception:
                pass
            finally:
                with _audio_lock:
                    if (self.wfile, client_lock) in _audio_clients:
                        _audio_clients.remove((self.wfile, client_lock))
        elif path=="/cmd_log":
            t=int(qs.get("tail","10")); recs=list(_cmd_ring)[-t:]
            st(json.dumps({"n_total":_n_cmd_total,"records":recs})+"\n")
        else: self.send_error(404)

    def _serve_panel(self):
        try:
            with open("/userdata/camera_stream/panel.html","rb") as f: h=f.read()
            self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(h)
        except: self.send_error(500,"panel.html not found")
    def _serve_jpeg(self):
        with _lock:
            if _latest_jpeg is None: self.send_error(503,"no frame"); return
            j=_latest_jpeg
        self.send_response(200); self.send_header("Content-Type","image/jpeg"); self.send_header("Content-Length",str(len(j))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(j)
    def _send_text(self,b):
        self.send_response(200); self.send_header("Content-Type","text/plain; charset=utf-8"); self.send_header("Content-Length",str(len(b))); self.end_headers()
        if isinstance(b,str): self.wfile.write(b.encode())
        else: self.wfile.write(b)
    def _send_json(self,d):
        j=json.dumps(d).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(j))); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers(); self.wfile.write(j)
    def _stream_mjpeg(self):
        self.send_response(200); self.send_header("Content-Type","multipart/x-mixed-replace; boundary=frame"); self.send_header("Cache-Control","no-store"); self.end_headers()
        try:
            while True:
                with _lock: j=_latest_jpeg
                if j: self.wfile.write(b"--frame\\r\\nContent-Type: image/jpeg\\r\\nContent-Length: %d\\r\\n\\r\\n" % len(j) + j + b"\\r\\n")
                self.wfile.flush(); time.sleep(0.1)
        except: pass
    def log_message(self, *a): pass

def _associate_targets(detections, tracked, max_dist):
    """贪心关联：将新检测到的目标与已有追踪目标按位置距离匹配。"""
    if not detections:
        for t in tracked:
            t['lost_frames'] += 1
        return [t for t in tracked if t['lost_frames'] <= TRACK_LOST_MAX]
    for t in tracked:
        t['lost_frames'] += 1
    unmatched = sorted(detections, key=lambda d: d['conf'], reverse=True)
    used = set()
    for det in unmatched:
        best_d = max_dist; best_i = -1
        for i, t in enumerate(tracked):
            if i in used: continue
            d = math.hypot(det['cx'] - t['cx'], det['cy'] - t['cy'])
            if d < best_d: best_d = d; best_i = i
        if best_i >= 0:
            t = tracked[best_i]
            t['cx'] = det['cx']; t['cy'] = det['cy']
            t['conf'] = det['conf']; t['cls'] = det['cls']
            t['bbox'] = det['bbox']; t['lost_frames'] = 0
            t['age'] += 1
            used.add(best_i)
        else:
            global _next_target_id
            tracked.append({
                'id': _next_target_id, 'cx': det['cx'], 'cy': det['cy'],
                'conf': det['conf'], 'cls': det['cls'], 'bbox': det['bbox'],
                'lost_frames': 0, 'age': 1,
            })
            _next_target_id += 1
    tracked = [t for t in tracked if t['lost_frames'] <= TRACK_LOST_MAX]
    tracked.sort(key=lambda t: t['conf'], reverse=True)
    return tracked[:MAX_TARGETS]

def _pid_output_stable(ex, ey):
    """PID 输出: ByteTrack 已 Kalman 平滑+预测, KP 保持固定 0.035.
    速率限制 + 最小输出置零"""
    global _last_yaw, _last_pitch
    yaw = ex * KP; pitch = -ey * KP
    yaw = max(-MAX_SPEED, min(MAX_SPEED, yaw))
    pitch = max(-MAX_SPEED, min(MAX_SPEED, pitch))
    if abs(yaw) < PID_SNAP: yaw = 0.0
    if abs(pitch) < PID_SNAP: pitch = 0.0
    yaw = np.clip(yaw, _last_yaw - RPM_RATE_LIMIT, _last_yaw + RPM_RATE_LIMIT)
    pitch = np.clip(pitch, _last_pitch - RPM_RATE_LIMIT, _last_pitch + RPM_RATE_LIMIT)
    _last_yaw, _last_pitch = yaw, pitch
    return yaw, pitch

def _compute_track_command(cx, cy):
    """状态机追踪: ByteTrack 已是 Kalman 平滑+预测, 这里只做状态机 + 死区.
    返回 (yaw_rpm, pitch_rpm, target_seen)"""
    global _track_state, _hold_counter
    ex = TARGET_X - cx; ey = TARGET_Y - cy
    if _track_state == STATE_IDLE:
        _track_state = STATE_TRACKING
        yaw, pitch = _pid_output_stable(ex, ey)
        return yaw, pitch, True
    elif _track_state == STATE_TRACKING:
        if abs(ex) < DEAD_ZONE_ENTER and abs(ey) < DEAD_ZONE_ENTER:
            _track_state = STATE_HOLDING
            _hold_counter = 0
            print(f"[track] HOLD enter (ex={ex:.0f}, ey={ey:.0f})")
            return 0.0, 0.0, True
        yaw, pitch = _pid_output_stable(ex, ey)
        return yaw, pitch, True
    else:  # STATE_HOLDING
        global _prev_ex, _prev_ey
        _hold_counter += 1
        # 速度检测: 目标持续远离中心则提前释放 (避免 EMA 慢响应导致 HOLD 卡死)
        vel_ex = ex - _prev_ex
        vel_ey = ey - _prev_ey
        _prev_ex, _prev_ey = ex, ey
        escaping = (abs(ex) > DEAD_ZONE_EXIT // 2 and ex * vel_ex > 0) or \
                   (abs(ey) > DEAD_ZONE_EXIT // 2 and ey * vel_ey > 0)
        if abs(ex) > DEAD_ZONE_EXIT or abs(ey) > DEAD_ZONE_EXIT or escaping:
            if _hold_counter >= HOLD_MIN_FRAMES:
                _track_state = STATE_TRACKING
                tag = "release(vel)" if escaping else "release"
                print(f"[track] HOLD {tag} (ex={ex:.0f}, ey={ey:.0f}, held={_hold_counter}fr)")
                yaw, pitch = _pid_output_stable(ex, ey)
                return yaw, pitch, True
        return 0.0, 0.0, True

def _laser_gen_heart(n=60):
    """生成归一化 [-1,1] 爱心曲线 waypoints (心尖朝下, t=π 处 waypoint pitch 落在正向最大值 -> 视觉下方).
    公式 y 取负, 因为 _laser_draw_pattern 里 pitch_rpm = -dpitch, 已是 math-up -> visual-down 的翻折."""
    pts = []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        x = 16 * math.sin(t) ** 3
        y_raw = 13 * math.cos(t) - 5*math.cos(2*t) - 2*math.cos(3*t) - math.cos(4*t)
        y = -y_raw  # 心尖位置 (原 t=π 处的最小值 -17) 翻为最大值 +17 -> norm_y=+1
        # normalize (max |x|~16, max |y|~17)
        pts.append((x / 16.0, y / 17.0))
    return pts

def _laser_gen_star():
    """生成 5 角星 (10 个顶点 + 闭合), 内/外半径比 0.382"""
    pts = []
    R, r_in = 1.0, 0.382
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5   # 从顶部开始, 每 36°
        r = R if i % 2 == 0 else r_in
        pts.append((r * math.cos(angle), r * math.sin(angle)))
    pts.append(pts[0])   # 闭合
    return pts

LASER_YAW_DEG = 30.0     # 图案覆盖的云台 yaw 总跨度
LASER_PITCH_DEG = 25.0   # pitch 总跨度
LASER_MARGIN = 0.80      # 留 20% 安全边距
LASER_DEFAULT_DUR = 8.0  # 默认绘制时长 (秒)

def _laser_draw_pattern(name, total_dur=LASER_DEFAULT_DUR):
    """驱动云台走完一个图案. 会临时关闭追踪避免抢云台."""
    global _laser_drawing, _manual_mode
    if _gim is None:
        print("[laser] no gimbal"); return False, "no_gimbal"
    if _laser_drawing:
        print("[laser] already drawing"); return False, "busy"
    if name == "heart":
        norm_pts = _laser_gen_heart(60)
    elif name == "star":
        norm_pts = _laser_gen_star()
    else:
        return False, "unknown_pattern"
    # 归一化坐标 -> 云台角度 (rad)
    yaw_max = LASER_YAW_DEG * LASER_MARGIN / 2
    pitch_max = LASER_PITCH_DEG * LASER_MARGIN / 2
    waypoints = [(p[0] * yaw_max, p[1] * pitch_max) for p in norm_pts]
    
    _laser_drawing = True
    # 暂时切手动模式, 防止 detection_loop 抢云台
    saved_manual = _manual_mode
    _manual_mode = True
    print(f"[laser] start pattern={name} pts={len(waypoints)} dur={total_dur}s yaw_max=±{yaw_max:.1f}° pitch_max=±{pitch_max:.1f}°")
    t0 = time.monotonic()
    try:
        n = len(waypoints)
        per_seg = total_dur / max(n - 1, 1)
        prev_yaw, prev_pitch = 0.0, 0.0
        for i, (yaw, pitch) in enumerate(waypoints):
            dyaw = yaw - prev_yaw
            dpitch = pitch - prev_pitch
            # 1 RPM = 6°/sec -> 需要的 RPM = 角度/(秒*6)
            yaw_rpm = dyaw / (per_seg * 6.0)
            pitch_rpm = -dpitch / (per_seg * 6.0)   # 屏幕 Y 向上为正, 云台 pitch 向上为正
            # 限速, 避免过冲
            yaw_rpm = max(-20.0, min(20.0, yaw_rpm))
            pitch_rpm = max(-20.0, min(20.0, pitch_rpm))
            _gim.send(yaw_rpm, pitch_rpm)
            time.sleep(per_seg)
            prev_yaw, prev_pitch = yaw, pitch
        # 停止
        _gim.send(0, 0)
        dur = time.monotonic() - t0
        print(f"[laser] done {name} actual_dur={dur:.2f}s")
        return True, f"done {dur:.1f}s"
    except Exception as e:
        print(f"[laser] FAIL {name}: {e}")
        try: _gim.send(0, 0)
        except: pass
        return False, str(e)
    finally:
        _manual_mode = saved_manual   # 恢复追踪
        _laser_drawing = False

def detect_loop():
    """独立 BPU 检测线程 (A1 拆分), 不阻塞 detection_loop.
    主循环每 N_DETECT 帧 set _detect_event. 本线程等事件 -> 跑 _det.detect (500-600ms) -> 写 _latest_bboxes -> 清 _detect_busy."""
    global _latest_bboxes, _last_bpu, _bpu_last_fps, _detect_busy, _bboxes_lock, _frame_lock, _latest_frame, _detect_pending_n
    bpu_c = 0
    bpu_t0 = time.monotonic()
    while True:
        _detect_event.wait()
        _detect_event.clear()
        with _bboxes_lock:
            if not _detect_busy:
                continue
            n_pending = _detect_pending_n
        # 用 _latest_frame 当前帧做 BPU (主循环每帧都在写最新 frame)
        with _frame_lock:
            f = _latest_frame
        if f is None:
            with _bboxes_lock:
                _detect_busy = False
            continue
        try:
            lb, r, dx, dy = letterbox(f, 640)
            nv12 = bgr_to_nv12(lb)
            pred = _det.detect(nv12)
            new_ob = [((x1-dx)/r, (y1-dy)/r, (x2-dx)/r, (y2-dy)/r, conf, int(cls))
                      for x1, y1, x2, y2, conf, cls in pred]
            with _bboxes_lock:
                _latest_bboxes = new_ob
                _detect_busy = False
                _last_bpu = n_pending
            bpu_c += 1
        except Exception as e:
            print(f"[detect] FAIL: {e}")
            with _bboxes_lock:
                _detect_busy = False
        if time.monotonic() - bpu_t0 >= 1.0:
            _bpu_last_fps = bpu_c / (time.monotonic() - bpu_t0)
            bpu_t0 = time.monotonic()
            bpu_c = 0

def render_loop():
    """独立 JPEG 渲染线程 (A1 拆分), 10fps 给前端 /frame.jpg.
    每 100ms 从 _latest_frame 拿最新 frame -> cv2.imencode JPEG -> 写 _latest_jpeg.
    不画任何 overlay (前端自己画)."""
    global _latest_jpeg, _frame_lock, _latest_frame, _lock
    while True:
        time.sleep(0.1)  # 10fps
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            continue
        try:
            ret, j = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ret:
                with _lock:
                    _latest_jpeg = j.tobytes()
        except Exception as e:
            print(f"[render] FAIL: {e}")

def detection_loop():
    """追踪主循环 (A1 拆分): 只做 cap.read + ByteTracker.update (空检测也调, 让 Kalman 预测) + PID.
    不画 overlay, 不编码 JPEG, 不调 BPU (BPU 异步跑在 detect_loop).
    共享 frame 给 render_loop, 共享 _latest_bboxes (从 detect_loop 来).
    频率 = cap 硬件上限 (30fps), 不再被 BPU 600ms 阻塞."""
    global _n_frame, _capture_actual_fps, _bpu_last_fps, _latest_frame, _latest_bboxes
    global _v4l2_recording, _ever_tracked, _track_state, _hold_counter, _ema_cx, _ema_cy, _ema_initialized, _last_yaw, _last_pitch
    global _last_bpu, _bboxes_lock, _detect_busy, _detect_pending_n, _detect_event
    sec_c = 0; sec_t0 = time.monotonic()
    _ever_tracked = False
    while True:
        if _v4l2_recording or _cap is None or not _cap.isOpened():
            time.sleep(0.05); continue
        ok, frame = _cap.read()
        if not ok: time.sleep(0.001); continue
        _n_frame += 1
        # 1) 共享 frame 给 render_loop + detect_loop
        with _frame_lock:
            global _latest_frame; _latest_frame = frame
        # 2) 每 N_DETECT 帧触发 BPU 异步 (只在 detect 闲时塞, 否则跳过本帧)
        if _n_frame - _last_bpu >= N_DETECT:
            with _bboxes_lock:
                if not _detect_busy:
                    _detect_pending_n = _n_frame
                    _detect_busy = True
                    _detect_event.set()
        # 3) 读最新检测结果 (主循环不依赖 BPU 完成)
        with _bboxes_lock:
            oboxes = list(_latest_bboxes) if _latest_bboxes else []
        # 4) ByteTracker 每帧 update (空检测也调, Kalman 预测填补)
        yaw_s, pitch_s = 0.0, 0.0; target_seen = False
        if _tracking and not _manual_mode and _byte_tracker is not None:
            filtered = []
            for x1, y1, x2, y2, conf, cls in oboxes:
                ci = int(cls)
                if 0 <= ci < len(COCO) and COCO[ci] in TRACK_CLASSES:
                    filtered.append((x1, y1, x2, y2, conf, cls))
            if filtered:
                dets_np = np.array(filtered, dtype=np.float32)
                dets_t = torch.from_numpy(dets_np)
            else:
                # 传空张量让 ByteTrack 内部走 Kalman 预测
                dets_t = torch.zeros((0, 6), dtype=torch.float32)
            tracks_out = _byte_tracker.update(dets_t, None)
            if len(tracks_out) > 0:
                best_idx = int(np.argmax(tracks_out[:, 6]))
                x1, y1, x2, y2, tid, cls, score = tracks_out[best_idx]
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                yaw_s, pitch_s, target_seen = _compute_track_command(cx, cy)
                if target_seen:
                    _ever_tracked = True
        if _manual_mode and check_manual_timeout(): print("[track] manual timeout->auto")
        if target_seen and not _manual_mode and _gim: _gim.send(yaw_s, pitch_s)
        elif not target_seen and not _manual_mode and _gim: _gim.send(0, 0)
        sec_c += 1
        if time.monotonic() - sec_t0 >= 1.0:
            _capture_actual_fps = sec_c / (time.monotonic() - sec_t0)
            sec_t0 = time.monotonic()
            sec_c = 0

def main():
    global _cap, _det, _gim
    _cap=cv2.VideoCapture(0,cv2.CAP_ANY); _cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*"MJPG"))
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W); _cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H); _cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not _cap.isOpened(): _cap=cv2.VideoCapture("/dev/video0",cv2.CAP_V4L2)
    print(f"[track] camera opened: /dev/video0 isOpened={_cap.isOpened()}")
    _det=Detector("/home/sunrise/yolov5/yolov5s_custom.bin"); _gim=None
    global _byte_tracker
    if _HAS_BYTETRACK:
        # frame_rate ≈ capture_fps 6 / N_DETECT 4 = 1.5, 给到 5 让 KF 时间步更合理
        _byte_tracker = BYTETracker(track_thresh=0.4, track_buffer=25, match_thresh=0.8, frame_rate=5)
        print(f"[track] ByteTracker ready (track_thresh=0.4, buffer=25fr, match=0.8)")
    try:
        import glob as _glob
        _ports=sorted(_glob.glob("/dev/ttyUSB*")+_glob.glob("/dev/ttyACM*"))
        _port=_ports[0] if _ports else "/dev/ttyUSB0"
        _gim=Gimbal(_port)
    except Exception as e:
        print(f"[gim] init failed: {e}, gimbal disabled")
        _gim=None
    # A1 拆分: 启动 3 个独立线程 (追踪 / 渲染 / BPU 异步)
    threading.Thread(target=detect_loop, daemon=True).start()
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=render_loop, daemon=True).start()
    # 关键: ThreadingHTTPServer 让 /audio_stream /video_feed 等长连接 handler 不阻塞 /frame.jpg /其它端点
    sv=ThreadingHTTPServer(("0.0.0.0",PORT),Handler)
    print(f"[track] yuntai.py PID: target=({TARGET_X},{TARGET_Y}) kp={KP} dz={DEAD_ZONE} max={MAX_SPEED}")
    print(f"[track] endpoints: / /health /track/on|off /target /ptz/manual /frame.jpg /video_feed /audio_stream /stats /cmd_log /record/start|stop|list /recordings/<file> /zoom /focus /aim /mode/manual /mode/auto /uvc/get /voice/list /voice/play /voice/ptt/start /voice/ptt/stop /voice/stt")
    try: sv.serve_forever()
    except KeyboardInterrupt: print("[track] exiting")

if __name__=="__main__": main()
