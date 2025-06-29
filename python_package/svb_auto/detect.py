from PIL import Image
import cv2
from hamcrest import is_
import numpy as np
import os
from typing import List, Union
from numpy.typing import NDArray
from .utils import crop_rectangle_relative

def detect_template_in_area(
        area_image: Union[NDArray, Image.Image], 
        template_image: Union[NDArray, Image.Image], 
        threshold=0.8,
        method=cv2.TM_CCOEFF_NORMED):
    """
    在指定区域中检测模板图像
    
    Args:
        area_image: PIL Image / Numpy Array - 搜索区域
        template_image: PIL Image / Numpy Array - 模板图像
        threshold: float - 匹配阈值 (0-1)
        meshod: int - OpenCV模板匹配方法 (默认使用 cv2.TM_CCORR)
    Returns:
        tuple: (is_found, confidence, location)

    """
    # 转换PIL图像为NumPy数组
    if isinstance(area_image, Image.Image):
        area_image = np.array(area_image)
    if isinstance(template_image, Image.Image):
        template_image = np.array(template_image)
    
    # 执行模板匹配
    result = cv2.matchTemplate(area_image, template_image, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 判断是否找到匹配
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        # 对于平方差方法，较小的值表示更好的匹配
        is_found = min_val <= threshold
        max_val = min_val
    else:
        is_found = max_val >= threshold
    
    return is_found, max_val, max_loc

# rectangle positions
# None means the whole screen
rects = {
    "money": (0.614, 0.05, 0.92, 0.1),
    "crystal": (0.614, 0.05, 0.92, 0.1),
    "app_icon": None,
    "starting": (0.017, 0.72, 0.056, 0.79),  # 用特殊区域来检测是不是处于开始界面
    # 私人对战检测区域，用于检测是否处于对战界面，随机对战按钮颜色变化较大
    "private_battle": (1170/1920, 701/1080, 1332/1920, 785/1080),
    "battle_start": (1490/1920, 529/1080, 1836/1920, 884/1080),  # 对战开始按钮区域
    "end": (958/1920, 787/1080, 1351/1920, 884/1080),  # 结束回合的确认按钮，在还有牌可以用时会弹出
    "envolve": (22/1920, 168/1080, 660/1920, 563/1080),  # 进化按钮区域
    "super_envolve": (22/1920, 168/1080, 660/1920, 563/1080),  # 超进化按钮区域
    'treasure_result': (828/1920, 40/1080, 1102/1920, 118/1080 ),  # 宝箱奖励结果位置，在有宝箱活动的时候检测到该结果则点击跳过
    "retry": (958/1920, 757/1080, 1351/1920, 884/1080),  # 网络不稳定时点击重试
    "can_envolve": (810/1920, 736/1080, 873/1920, 800/1080),  # 检测是否可以进化
    "can_super_envolve": (1035/1920, 736/1080, 1115/1920, 800/1080),  # 检测是否可以超进化
    "return_to_battle": (860/1920, 192/1080, 1063/1920, 255/1080),  # 返回对战窗口标题，闪退再打开会出现该窗口
}

class Detector:
    def __init__(self, img_dir: str = "imgs_chs_1920_1080"):
        self.all_templates = {}
        self.templates_size = {}
        for fname in os.listdir(img_dir):
            fpath = os.path.join(img_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                img = Image.open(fpath).convert('RGB')
                key = os.path.splitext(fname)[0]
                self.all_templates[key] = np.array(img)
                self.templates_size[key] = img.size
    
    def detect_on_screen(
        self,
        screen_shot: Image.Image,
        template_name: str,
        threshold: float = 0.8,
        method=cv2.TM_CCOEFF_NORMED
    ) -> tuple[bool, float, tuple[int, int]]:

        img_width, img_height = screen_shot.size
        area = rects.get(template_name)

        # 获取待检测区域，以及该区域在屏幕上所在位置（一个矩形），
        # 该矩形中心会作为返回值一部分
        if area is None:
            # 如果 area 为 None，则表示检测整个屏幕
            # 这种情况一般是检测应用图标等全屏元素
            area_image = screen_shot
            rectangle = (0, 0, img_width, img_height)
        else:
            area_image = crop_rectangle_relative(
                screen_shot, 
                area
            )
            rectangle = (
                int(area[0] * img_width), 
                int(area[1] * img_height), 
                int(area[2] * img_width), 
                int(area[3] * img_height)
            )
        template_img = self.all_templates.get(template_name)
        template_size = self.templates_size.get(template_name)
        if template_img is None:
            raise ValueError(f"Template '{template_name}' not found in the provided directory.")
        is_detected, value, position = detect_template_in_area(
            area_image=area_image,
            template_image=template_img,
            threshold=threshold,
            method=method
        )
        position = (
            rectangle[0] + position[0] + template_size[0] // 2,
            rectangle[1] + position[1] + template_size[1] // 2
        )
        return is_detected, value, position