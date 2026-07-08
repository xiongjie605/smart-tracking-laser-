# RDK X5 智能追踪与光影交互系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RDK%20X5-orange)](https://www.dorabot.com/)

> 基于地瓜机器人 RDK X5 开发的嵌入式智能视觉追踪系统，集成 BPU 加速 YOLO 检测、三轴云台跟随、MJPEG 直播推流、紫外线激光投影及双向语音对讲，适用于宠物陪伴、互动直播与智能家居场景。

---

## ✨ 主要特性

- 🎯 **视觉追踪**：利用 BPU (10 TOPS) 加速 YOLOv5s 检测，结合 ByteTrack 多目标追踪算法，稳定锁定人/猫/狗，驱动云台保持目标居中。
- 📡 **实时推流**：MJPEG 视频流叠加追踪框，端到端延迟 <500ms，手机/电脑浏览器即可观看。
- 🎨 **激光投影**：通过云台带动紫外线激光模组，在荧光纸上绘制爱心、五角星等预设图案（打赏互动利器）。
- 🎤 **双向语音**：支持实时环境收音与按住说话 (PTT)，集成 SenseVoice 离线语音识别，无需联网。
- 🎬 **视频录制**：调用 FFmpeg 录制 MP4（含音频），可通过网页一键下载回放。
- 🕹️ **双模式控制**：直播模式（侧重打赏联动）/ 智能家居模式（侧重手动云台与对讲）。
- ⚙️ **多任务并发**：单板卡同时运行采集、推理、推流、录音、录制，线程隔离数据同步。

---

## 🖥️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         RDK X5                             │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ 摄像头   │  │   BPU    │  │  HTTP    │  │  音频    │   │
│  │ 采集    │─▶│ YOLOv5s  │  │ 服务端   │  │  处理    │   │
│  └─────────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│                    │              │              │         │
│           ┌────────▼──────────┐  │              │         │
│           │  ByteTrack + PID  │  │              │         │
│           │  (卡尔曼滤波)     │  │              │         │
│           └────────┬──────────┘  │              │         │
│                    │             │              │         │
│              ┌─────▼─────┐       │              │         │
│              │ 串口云台   │       │              │         │
│              └─────┬─────┘       │              │         │
└────────────────────┼─────────────┼──────────────┼─────────┘
                     │             │              │
          ┌──────────▼──────────┐  │       ┌──────▼──────┐
          │  三轴伺服云台       │  │       │ USB 麦克风  │
          │  (Yaw/Pitch/Roll)   │  │       │ / 扬声器    │
          └─────────────────────┘  │       └─────────────┘
                                   │
          ┌────────────────────────▼─────────┐
          │      紫外线激光模组              │
          │      (GPIO/PWM 控制)             │
          └──────────────────────────────────┘
```

---

## 🛠️ 硬件清单

| 组件 | 说明 |
|-----------|-------------|
| **RDK X5** | 主控，含 Bayes BPU (10 TOPS)，8GB 内存，Wi-Fi 6 |
| **USB 摄像头** | 免驱，支持 MJPEG 输出（推荐 640×360） |
| **三轴云台** | 无刷伺服 + BGC 控制板 (STM32F103)，UART 通信 (115200, 8N1) |
| **紫外线激光** | 405nm 激光模组，PWM/GPIO 控制 |
| **USB 声卡** | 集成麦克风与扬声器，16kHz 单声道采集 |
| **电源** | 云台 12V/3A，RDK X5 独立供电 |

---

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/xiongjie605/rdk-x5-tracker.git
cd rdk-x5-tracker
```

### 2. 连接外设
- USB 摄像头 → RDK X5 USB 3.0 口
- USB 声卡 → 另一 USB 口
- 云台 UART → GPIO (TXD/RXD)
- 激光模组 → GPIO/PWM 引脚

### 3. 安装依赖（在 RDK X5 上）
```bash
pip install -r requirements.txt
```

### 4. 运行服务
```bash
# 开发调试模式
python3 track_server.py

# 或安装为系统自启服务（推荐）
sudo ./install-service.sh
```

启动后，在浏览器访问 `http://<RDK_X5_IP>:8080` 即可看到控制面板。

---

## 📡 API 接口（核心）

所有接口默认监听 `8080` 端口。

| 接口 | 方法 | 功能 |
|----------|--------|-------------|
| `/` | GET | 控制面板页面 |
| `/frame.jpg` | GET | 获取最新 JPEG 帧（轮询） |
| `/video_feed` | GET | MJPEG 视频流 |
| `/audio_stream` | GET | 实时 PCM 音频流 (16kHz) |
| `/track/on` | GET | 开启自动追踪 |
| `/track/off` | GET | 关闭自动追踪 |
| `/mode/manual` | GET | 切换手动模式（永不超时） |
| `/mode/auto` | GET | 恢复自动追踪 |
| `/aim?yaw=<度>&pitch=<度>` | GET | 手动相对转动云台 |
| `/zoom?value=<0-260>` | GET | 数字变焦（摄像头支持） |
| `/record/start?auto=<秒>` | GET | 开始录制（自动停止） |
| `/record/stop` | GET | 停止录制 |
| `/recordings/<文件名>` | GET | 下载录制视频 |
| `/laser/pattern/heart` | GET | 绘制爱心图案 |
| `/laser/pattern/star` | GET | 绘制五角星 |
| `/laser/stop` | GET | 紧急停止激光 |
| `/voice/ptt/start` | GET | 开始按住说话录音 |
| `/voice/ptt/stop` | GET | 停止录音并自动播放 |
| `/audio/volume/set?value=<0-100>` | GET | 设置扬声器音量 |

---

## 🧠 核心算法说明

- **检测与追踪**：YOLOv5s 在 BPU 上推理（每 4 帧一次），ByteTrack 利用卡尔曼滤波与 IoU 匹配进行帧间关联，分配稳定 ID。
- **云台防抖**：采用带迟滞的死区（进入 14px / 退出 22px）与状态机（TRACKING ↔ HOLDING），避免目标静止时云台微颤。
- **激光绘制**：将图案离散为路径点（爱心 61 点），通过云台速度插值实现 8 秒连续平滑绘制。
- **音频降噪**：噪声门 + 高通/低通/陷波滤波器 + AGC（自动增益控制），消除电机嗡声与 USB 时钟尖峰。

---

## 📁 项目结构

```
rdk-x5-tracker/
├── track_server.py          # 主服务程序（一体化）
├── panel.html               # 双模式控制界面
├── install-service.sh       # Systemd 自启安装脚本
├── requirements.txt         # Python 依赖
├── LICENSE                  # MIT 协议
├── README.md                # 英文说明
└── README_CN.md             # 中文说明（本文件）
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。请遵循标准 GitHub Flow：
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/xxx`)
3. 提交修改
4. 推送到分支
5. 创建 Pull Request

---

## 📄 许可证

本项目采用 **MIT 许可证**，详见 [LICENSE](LICENSE) 文件。

---

**祝开发顺利！** 🚀
