import re
from PIL import Image
import cv2
from hamcrest import is_
import numpy as np
import os
from typing import List, Union, Tuple
from numpy.typing import NDArray
from regex import template
from .utils import crop_rectangle_relative, crop_rectangle

def detect_template_in_area(
        area_image: Union[NDArray, Image.Image], 
        template_image: Union[NDArray, Image.Image], 
        threshold=0.8,
        mask: NDArray= None,
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
    result = cv2.matchTemplate(area_image, template_image, method, mask=mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 判断是否找到匹配
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        # 对于平方差方法，较小的值表示更好的匹配
        is_found = min_val <= threshold
        max_val = min_val
    else:
        is_found = max_val >= threshold
    
    return is_found, max_val, max_loc

def detect_template_in_area_multi(
        area_image: Union[NDArray, Image.Image], 
        template_image: Union[NDArray, Image.Image], 
        threshold=0.8,
        mask: NDArray = None,
        method=cv2.TM_CCOEFF_NORMED,
        overlap_threshold=0.5):
    """
    在指定区域中检测所有匹配的模板图像位置
    
    Args:
        area_image: PIL Image / Numpy Array - 搜索区域
        template_image: PIL Image / Numpy Array - 模板图像
        threshold: float - 匹配阈值 (0-1)
        mask: NDArray - 遮罩图像
        method: int - OpenCV模板匹配方法 (默认使用 cv2.TM_CCOEFF_NORMED)
        overlap_threshold: float - 重叠阈值，用于非极大值抑制 (0-1)
    Returns:
        list: [(confidence, location), ...] - 所有匹配结果的列表
    """
    # 转换PIL图像为NumPy数组
    if isinstance(area_image, Image.Image):
        area_image = np.array(area_image)
    if isinstance(template_image, Image.Image):
        template_image = np.array(template_image)
    
    # 获取模板尺寸
    template_height, template_width = template_image.shape[:2]
    
    # 执行模板匹配
    result = cv2.matchTemplate(area_image, template_image, method, mask=mask)
    
    # 根据匹配方法确定阈值判断逻辑
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        # 对于平方差方法，较小的值表示更好的匹配
        locations = np.where(result <= threshold)
        confidences = result[locations]
    else:
        # 对于其他方法，较大的值表示更好的匹配
        locations = np.where(result >= threshold)
        confidences = result[locations]
    
    # 将位置和置信度组合
    matches = []
    for i in range(len(locations[0])):
        y, x = locations[0][i], locations[1][i]
        confidence = confidences[i]
        matches.append((confidence, (x, y)))
    
    # 如果没有匹配结果，直接返回空列表
    if not matches:
        return []
    
    # 使用非极大值抑制来去除重叠的检测结果
    filtered_matches = _non_max_suppression(
        matches, 
        template_width, 
        template_height, 
        overlap_threshold,
        method
    )
    
    return filtered_matches


def _non_max_suppression(matches, template_width, template_height, overlap_threshold, method):
    """
    非极大值抑制，用于去除重叠的检测结果
    
    Args:
        matches: list - [(confidence, (x, y)), ...] 匹配结果列表
        template_width: int - 模板宽度
        template_height: int - 模板高度  
        overlap_threshold: float - 重叠阈值
        method: int - OpenCV模板匹配方法
    Returns:
        list: 过滤后的匹配结果
    """
    if not matches:
        return []
    
    # 根据置信度排序（对于平方差方法，升序；其他方法，降序）
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        matches.sort(key=lambda x: x[0])  # 升序，值越小越好
    else:
        matches.sort(key=lambda x: x[0], reverse=True)  # 降序，值越大越好
    
    filtered_matches = []
    
    for current_match in matches:
        current_confidence, (current_x, current_y) = current_match
        
        # 检查当前匹配是否与已接受的匹配重叠
        is_overlapping = False
        for accepted_confidence, (accepted_x, accepted_y) in filtered_matches:
            # 计算重叠区域
            overlap_area = _calculate_overlap(
                current_x, current_y, template_width, template_height,
                accepted_x, accepted_y, template_width, template_height
            )
            
            # 计算重叠比例
            template_area = template_width * template_height
            overlap_ratio = overlap_area / template_area
            
            if overlap_ratio > overlap_threshold:
                is_overlapping = True
                break
        
        # 如果不重叠，则添加到结果中
        if not is_overlapping:
            filtered_matches.append(current_match)
    
    return filtered_matches


def _calculate_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    """
    计算两个矩形的重叠面积
    
    Args:
        x1, y1, w1, h1: 第一个矩形的位置和尺寸
        x2, y2, w2, h2: 第二个矩形的位置和尺寸
    Returns:
        int: 重叠面积
    """
    # 计算重叠区域的边界
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    
    # 如果没有重叠，返回0
    if left >= right or top >= bottom:
        return 0
    
    return (right - left) * (bottom - top)

# rectangle positions
# None means the whole screen
rects = {
    "money": (0.614, 0.05, 0.92, 0.1),
    "crystal": (0.614, 0.05, 0.92, 0.1),
    "app_icon": None,
    "starting": (0.017, 0.72, 0.056, 0.79),  # 用特殊区域来检测是不是处于开始界面
    # 私人对战检测区域，用于检测是否处于对战界面，随机对战按钮颜色变化较大
    "private_battle": (1100/1920, 600/1080, 1340/1920, 800/1080),
    "battle_start": (1490/1920, 529/1080, 1836/1920, 884/1080),  # 对战开始按钮区域
    "end": (958/1920, 787/1080, 1351/1920, 884/1080),  # 结束回合的确认按钮，在还有牌可以用时会弹出
    "envolve": (22/1920, 168/1080, 660/1920, 563/1080),  # 进化按钮区域
    "super_envolve": (22/1920, 168/1080, 660/1920, 563/1080),  # 超进化按钮区域
    'treasure_result': (828/1920, 40/1080, 1102/1920, 118/1080 ),  # 宝箱奖励结果位置，在有宝箱活动的时候检测到该结果则点击跳过
    "retry": (958/1920, 757/1080, 1351/1920, 884/1080),  # 网络不稳定时点击重试
    "can_envolve": (810/1920, 736/1080, 873/1920, 800/1080),  # 检测是否可以进化
    "can_super_envolve": (1035/1920, 736/1080, 1115/1920, 800/1080),  # 检测是否可以超进化
    "return_to_battle": (860/1920, 192/1080, 1063/1920, 255/1080),  # 返回对战窗口标题，闪退再打开会出现该窗口
    "hand_cards": (400/1920, 835/1080, 1600/1920, 945/1080),  # 手牌区域，检测手牌是否有可用的随从
    "field_opponent": (250/1920, 208/1080, 1540/1920, 470/1080),  # 对手场上随从区域
    "field_player": (250/1920, 470/1080, 1540/1920, 735/1080),  # 玩家场上随从区域
    "attack_opponent": (250/1920, 208/1080, 1540/1920, 470/1080),  # 对手随从攻击力区域
    "attack_player": (250/1920, 470/1080, 1540/1920, 735/1080),  # 玩家随从攻击力区域
    "ward_masked": None, # 用于检测随从是否处于守护状态，该 template 并非用于模板匹配，而是用于颜色特征匹配
}

# 背景颜色用于获得图片的遮罩 mask，从而能够使用带 mask 的模板匹配
bg_colors = {
    "attack_opponent": np.array([0, 255, 0], dtype=np.uint8),  # 随从攻击力背景值
    "attack_player": np.array([0, 255, 0], dtype=np.uint8),  # 随从攻击力背景值
    "ward_masked": np.array([0, 0, 0], dtype=np.uint8),  # 随从守护状态背景值
}

# 随从的位置的检测基于 "attack_opponent" 和 "attack_player" 的模板匹配
follower_margin = (0/1920, 155/1080, 220/1920, 100/1080)
follower_offset = (0, -155/1080)  # 随从位置相对于 "attack_opponent" 和 "attack_player" 检测位置的偏移量

# 随从守护状态的颜色范围,
# 直接在 RGB 空间里面的长方体定义颜色范围也许不是个很科学的主意，
# 但目前看来效果还不错
WARD_RED_RANGE = (159, 236)
WARD_GREEN_RANGE = (213, 255)
WARD_BLUE_RANGE = (69, 161)
WARD_COLOR_RATIO = 0.3 # 如果在护盾检测范围内，位于颜色范围内的像素占比超过该值，则认为是守护状态

class FollowerDetected:
    center_x: int
    center_y: int
    attack: int
    defense: int
    is_ward: bool = False
    # 当前默认所有随从都可以进行攻击，攻击之后都变成 False
    # 理想情况下：
    # - 攻击能力应当通过识别获得，且部分随从有连击能力
    can_attack_follower: bool = True  # 是否可以攻击随从
    can_attack_player: bool = True  # 是否可以攻击玩家
    def __init__(self, 
                 center_x: int, 
                 center_y: int, 
                 attack: int, 
                 defense: int, 
                 is_ward: bool = False):
        self.center_x = center_x
        self.center_y = center_y
        self.attack = attack
        self.defense = defense
        self.is_ward = is_ward

class Detector:
    def __init__(self, img_dir: str = "imgs_chs_1920_1080"):
        self.all_templates = {}
        self.templates_size = {}
        self.all_masks = {}
        for fname in os.listdir(img_dir):
            fpath = os.path.join(img_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                img = Image.open(fpath).convert('RGB')
                key = os.path.splitext(fname)[0]
                self.all_templates[key] = np.array(img)
                self.templates_size[key] = img.size
                if key in bg_colors:
                    # 如果有背景颜色定义，则创建遮罩 mask
                    bg_color = bg_colors[key]
                    mask = cv2.inRange(np.array(img), bg_color, bg_color)
                    self.all_masks[key] = cv2.bitwise_not(mask)
    
    def detect_on_screen(
        self,
        screen_shot: Image.Image,
        template_name: str,
        threshold: float = 0.8,
        method=cv2.TM_CCOEFF_NORMED,
    ) -> tuple[bool, float, tuple[int, int]]:
        """
        在屏幕截图中检测指定模板图像
        Args:
            screen_shot: PIL Image - 屏幕截图
            template_name: str - 模板图像的名称
            threshold: float - 匹配阈值 (0-1)
            method: int - OpenCV模板匹配方法 (默认使用 cv2.TM_CCOEFF_NORMED)
        Returns:
            tuple: (is_detected, max_value, position)
        
        """

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
            method=method,
            mask=self.all_masks.get(template_name, None)
        )
        
        position = (
            rectangle[0] + position[0] + template_size[0] // 2,
            rectangle[1] + position[1] + template_size[1] // 2
        )
        return is_detected, value, position
    
    
    def detect_followers(
            self,
            screen_shot: Image.Image,
            field_name: str = 'field_opponent',
        ) -> Tuple[List[FollowerDetected], List[Tuple[int, int, int, int]]]:
        """
        检测对手场上所有随从的信息
        """
        img_width, img_height = screen_shot.size
        area = rects[field_name]
        template_width, template_height = self.templates_size['ward_masked']
        area_image = crop_rectangle_relative(
            screen_shot, 
            area
        )

        attack_template = self.all_templates['attack_opponent']
        mask = self.all_masks['attack_opponent']
        detect_result = detect_template_in_area_multi(
            area_image=area_image,
            template_image=attack_template,
            threshold=0.8,
            method=cv2.TM_CCOEFF_NORMED,
            mask=mask
        )
        # 保证返回的尺寸和 ward_masked 的尺寸相同
        # @TODO 应当设定标准的随从尺寸
        detect_corners = [
            (result[1][0] + int((follower_offset[0]+area[0]) * img_width),
            result[1][1] + int((follower_offset[1]+area[1]) * img_height))
            for result in detect_result
        ]
        detect_rectangles = [
            (corner[0], corner[1],
             corner[0] + template_width, corner[1] + template_height)
            for corner in detect_corners
        ]

        followers = [
            FollowerDetected(
                center_x=int((rect[0] + rect[2]) // 2), # fix Object of type int64 is not JSON serializable
                center_y=int((rect[1] + rect[3]) // 2),
                attack=0,  # 攻击力待后续检测
                defense=0,  # 防御力待后续检测
                is_ward=self._is_follower_ward(
                    crop_rectangle(screen_shot, rect))
                )
            for rect in detect_rectangles
        ]
        return followers, detect_rectangles
    
    def _is_follower_ward(
      self,
      follower_img: Union[NDArray, Image.Image],
    ):
        # follower_img 的尺寸必须与 ward_masked 的尺寸相同
        # 这在 1920x1080 分辨率下目前可以保证，但是其他分辨率存疑
        if isinstance(follower_img, Image.Image):
            follower_img = np.array(follower_img)
        ward_mask = self.all_masks['ward_masked'].astype(bool)
        color_mask = np.ones(follower_img.shape[:2], dtype=bool)
        color_mask &= (follower_img[:, :, 0] >= WARD_RED_RANGE[0]) & (follower_img[:, :, 0] <= WARD_RED_RANGE[1])
        color_mask &= (follower_img[:, :, 1] >= WARD_GREEN_RANGE[0]) & (follower_img[:, :, 1] <= WARD_GREEN_RANGE[1])
        color_mask &= (follower_img[:, :, 2] >= WARD_BLUE_RANGE[0]) & (follower_img[:, :, 2] <= WARD_BLUE_RANGE[1])
        # 计算颜色范围内的像素占比
        color_mask = color_mask & ward_mask
        color_ratio = color_mask.sum() / ward_mask.sum()
        return color_ratio > WARD_COLOR_RATIO