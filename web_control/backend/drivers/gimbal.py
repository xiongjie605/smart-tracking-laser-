"""
云台驱动：抽自 yuntai.py (云台视觉追踪控制器)。
协议: 9 字节帧 0xAA 0x51 [yh][yl] 0 0 [ph][pl] 0x55, 115200 baud;
速度编码 rpm*100 int16 BE。
PID: kp=0.035, dead_zone=1, max_speed=26.0 (来自 yuntai.py 默认)。
"""
import serial


class Gimbal:
    BAUD = 115200
    KP = 0.035
    DEAD_ZONE = 1
    MAX_SPEED = 26.0

    def __init__(self, port: str = '/dev/ttyUSB0'):
        self.ser = None
        self._open(port)

    def _open(self, port: str):
        self.ser = serial.Serial(port, self.BAUD, timeout=0.1)

    @staticmethod
    def _speed_to_bytes(rpm: float):
        v = int(round(rpm * 100))
        v = max(-32768, min(32767, v))
        if v < 0:
            v += 65536
        return (v >> 8) & 0xFF, v & 0xFF

    def send(self, yaw: float, pitch: float) -> None:
        """下发 yaw/pitch 速度 (rpm)。0,0 即停止。"""
        if self.ser is None:
            return
        yh, yl = self._speed_to_bytes(yaw)
        ph, pl = self._speed_to_bytes(pitch)
        try:
            self.ser.write(bytearray([0xAA, 0x51, yh, yl, 0, 0, ph, pl, 0x55]))
        except Exception:
            pass

    def track(self, ex: float, ey: float, target_x: float = 320, target_y: float = 240) -> bool:
        """PID 跟踪: 像素误差 ex/ey, 中心 target_x,target_y。死区内返回 False。"""
        if abs(ex) < self.DEAD_ZONE and abs(ey) < self.DEAD_ZONE:
            self.send(0, 0)
            return False
        yaw = max(-self.MAX_SPEED, min(self.MAX_SPEED, ex * self.KP))
        pitch = max(-self.MAX_SPEED, min(self.MAX_SPEED, -ey * self.KP))
        self.send(yaw, pitch)
        return True

    def stop(self):
        self.send(0, 0)

    def close(self):
        try:
            self.stop()
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None
