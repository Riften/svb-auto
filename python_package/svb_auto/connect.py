from adbutils import adb
import uiautomator2 as u2
from adbutils import AdbClient

def connect_with_adbutils(port=16384):
    """
    连接到 ADB 设备并返回 u2 设备对象。
    
    :param port: ADB 服务器端口，默认为 5037。

    Mumu 模拟器端口查看：右上角三条线→问题诊断→往下滑有个网络信息→adb调试端口
    """
    
    devices = adb.device(f"127.0.0.1:{port}")

    if not devices:
        raise RuntimeError("请检查 USB 调试是否开启")

    u2_device = u2.connect(devices.serial)
    return u2_device 