# RDK X5 Smart Tracking & Light Interaction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RDK%20X5-orange)](https://www.dorabot.com/)

> An embedded visual tracking system based on the DoraRobot RDK X5, featuring BPU‑accelerated YOLO detection, 3‑axis gimbal tracking, MJPEG live streaming, UV laser projection, and two‑way voice. Designed for pet companions, interactive live streaming, and smart home scenarios.

---

## ✨ Features

- 🎯 **Visual Tracking** – YOLOv5s accelerated by RDK X5's BPU (10 TOPS) + ByteTrack multi‑target tracking. Locks onto person/cat/dog and drives a 3‑axis gimbal to keep the target centered.
- 📡 **Real‑time Streaming** – MJPEG stream with tracking overlays, end‑to‑end latency <500 ms, accessible via browser on any device.
- 🎨 **Laser Projection** – Draw preset patterns (heart, star, etc.) on fluorescent paper by commanding the gimbal to move a UV laser module – ideal for interactive gift feedback.
- 🎤 **Two‑way Voice** – Live audio broadcasting and push‑to‑talk (PTT) with offline STT (SenseVoice) – no cloud dependency.
- 🎬 **Video Recording** – Record MP4 with synchronized audio via FFmpeg; download recordings through the web interface.
- 🕹️ **Dual‑mode Control** – "Live" mode for interactive gifting; "Home" mode for manual gimbal control and voice intercom.
- ⚙️ **Multi‑task Concurrency** – All modules (camera capture, BPU inference, HTTP server, audio, recording) run in parallel with thread‑safe shared memory.

---

## 🖥️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         RDK X5                             │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Camera  │  │   BPU    │  │  HTTP    │  │  Audio   │   │
│  │Capture  │─▶│ YOLOv5s  │  │ Server   │  │  Stack   │   │
│  └─────────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│                    │              │              │         │
│           ┌────────▼──────────┐  │              │         │
│           │  ByteTrack + PID  │  │              │         │
│           │  (Kalman filter)  │  │              │         │
│           └────────┬──────────┘  │              │         │
│                    │             │              │         │
│              ┌─────▼─────┐       │              │         │
│              │  Serial   │       │              │         │
│              │  Gimbal   │       │              │         │
│              └─────┬─────┘       │              │         │
└────────────────────┼─────────────┼──────────────┼─────────┘
                     │             │              │
          ┌──────────▼──────────┐  │       ┌──────▼──────┐
          │  3‑axis Servo Gimbal│  │       │ USB Mic/   │
          │  (Yaw/Pitch/Roll)   │  │       │ Speaker    │
          └─────────────────────┘  │       └─────────────┘
                                   │
          ┌────────────────────────▼─────────┐
          │      UV Laser Module             │
          │  (GPIO/PWM controlled)           │
          └──────────────────────────────────┘
```

---

## 🛠️ Hardware Requirements

| Component | Description |
|-----------|-------------|
| **RDK X5** | Magicbox with Bayes BPU (10 TOPS), 8GB LPDDR4, Wi‑Fi 6 |
| **USB Camera** | UVC‑compliant, MJPEG output (640×360 recommended) |
| **3‑axis Gimbal** | Brushless servo with BGC controller (STM32F103) – communicates via UART (115200, 8N1) |
| **UV Laser Module** | 405nm laser with PWM/GPIO control |
| **USB Audio Card** | Integrated mic + speaker, 16kHz mono capture |
| **Power Supply** | 12V/3A for gimbal; RDK X5 via its own adapter |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/xiongjie605/rdk-x5-tracker.git
cd rdk-x5-tracker
```

### 2. Connect peripherals
- USB camera → RDK X5 USB 3.0 port
- USB audio card → another USB port
- Gimbal UART → GPIO pins (TXD/RXD)
- Laser module → GPIO/PWM pin

### 3. Install dependencies (on RDK X5)
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
# Development mode
python3 track_server.py

# Or install as a system service (recommended)
sudo ./install-service.sh
```

Open `http://<RDK_X5_IP>:8080` in your browser to access the control panel.

---

## 📡 API Endpoints

All endpoints are served on port `8080`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web control panel (HTML) |
| `/frame.jpg` | GET | Latest JPEG frame (polling) |
| `/video_feed` | GET | MJPEG streaming (multipart) |
| `/audio_stream` | GET | Live PCM audio stream (16kHz mono) |
| `/track/on`, `/track/off` | GET | Enable/disable auto tracking |
| `/mode/manual` | GET | Switch to manual gimbal control (no time‑out) |
| `/mode/auto` | GET | Resume auto tracking |
| `/aim?yaw=<deg>&pitch=<deg>` | GET | Manually aim the gimbal (relative angle) |
| `/zoom?value=<0-260>` | GET | Set camera digital zoom (if supported) |
| `/record/start?auto=<seconds>` | GET | Start video recording (auto‑stop after seconds) |
| `/record/stop` | GET | Stop recording |
| `/record/list` | GET | List recorded files |
| `/recordings/<filename>` | GET | Download recorded MP4 |
| `/laser/pattern/heart` (or `star`) | GET | Draw a laser pattern |
| `/laser/stop` | GET | Emergency stop laser drawing |
| `/voice/ptt/start` | GET | Begin push‑to‑talk recording |
| `/voice/ptt/stop` | GET | Stop PTT, auto‑play back through speaker, return JSON |
| `/voice/stt` | POST | Send WAV file for offline STT (SenseVoice) |
| `/audio/volume/set?value=<0-100>` | GET | Set speaker volume |
| `/stats` | GET | JSON with current FPS, tracking status, etc. |

---

## 🧠 Core Algorithms

- **Detection & Tracking** – YOLOv5s runs on BPU (every 4 frames). ByteTrack uses Kalman filters and IoU matching for frame‑to‑frame association, assigning persistent IDs.
- **Gimbal Smoothing** – A dead‑zone with hysteresis (enter=14px, exit=22px) and a state machine (TRACKING ↔ HOLDING) prevent micro‑jitter when the target is nearly stationary.
- **Laser Drawing** – Patterns are discretized into waypoints (e.g., 61 points for a heart). The gimbal moves smoothly between points via speed interpolation – total drawing time ~8 seconds.
- **Audio Pipeline** – Noise gate + HPF/LPF/notch filters + AGC (automatic gain control) remove motor hum and USB clock noise. PTT recordings are saved as WAV and auto‑played via `aplay`; offline STT uses SenseVoice.

---

## 📁 Project Structure

```
rdk-x5-tracker/
├── track_server.py          # Main server (all‑in‑one)
├── panel.html               # Web UI (dual‑mode)
├── install-service.sh       # Systemd installation script
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── README.md                # English version (this file)
└── README_CN.md             # Chinese version
```

---

## 🤝 Contributing

We welcome issues and pull requests. Please follow the standard GitHub flow:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

**Happy Building!** 🚀
