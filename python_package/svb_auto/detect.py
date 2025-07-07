from PIL import Image
import cv2
import numpy as np
import os
from typing import List, Union, Tuple
from numpy.typing import NDArray
from .utils import crop_rectangle_relative, crop_rectangle, debug_draw_rectangles

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

def combined_color_canny_matching(template_border_color, scene_orig,
                                  canny_low_threshold=50, canny_high_threshold=150,
                                  match_threshold=0.7):
    """
    Args:
        template_border_color (np.array): shield image array
        scene_orig (np.array): target scene image array
        canny_low_threshold (int): Canny 边缘检测的低阈值。
        canny_high_threshold (int): Canny 边缘检测的高阈值。
        match_threshold (float): 模板匹配的相似度阈值(0.0-1.0)。
    Returns:
        tuple: (is_found, confidence, location)
    """
    # Compare under hsv
    scene_hsv = cv2.cvtColor(scene_orig, cv2.COLOR_BGR2HSV)

    lower_green_hsv_threshold = np.array([30, 100, 210])
    upper_green_hsv_threshold = np.array([55, 160, 255])


    green_mask = cv2.inRange(scene_hsv, lower_green_hsv_threshold, upper_green_hsv_threshold)

    scene_gray = cv2.cvtColor(scene_orig, cv2.COLOR_BGR2GRAY)
    scene_canny_edges = cv2.Canny(scene_gray, canny_low_threshold, canny_high_threshold)

    combined_scene_edges = cv2.bitwise_and(scene_canny_edges, scene_canny_edges, mask=green_mask)
    
    template_border_gray = cv2.cvtColor(template_border_color, cv2.COLOR_BGR2GRAY)
    template_edges = cv2.Canny(template_border_gray, canny_low_threshold, canny_high_threshold)

    result = cv2.matchTemplate(combined_scene_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    return max_val > match_threshold, max_val, max_loc


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
    "group_stage": (1034 / 1920, 426 / 1080, 1148 / 1920, 540 / 1080),
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
# WARD_GREEN_RANGE = (213, 255)
# WARD_BLUE_RANGE = (69, 161)
WARD_GREEN_RANGE = (180, 255)
WARD_BLUE_RANGE = (50, 175)
WARD_COLOR_RATIO = 0.25 # 如果在护盾检测范围内，位于颜色范围内的像素占比超过该值，则认为是守护状态

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
        self.feature_matcher = FeatureMatcher(rects=rects)
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
    
    def debug_draw_all_ward(self, screen: Image.Image):
        """
        在屏幕截图上绘制所有随从的守护状态
        """
        screen_copy = screen.copy()
        followers, rectangles = self.detect_followers(screen_copy, field_name='field_player')
        follower_opponent, rectangles_opponent = self.detect_followers(screen_copy, field_name='field_opponent')
        for i in range(len(followers)):
            follower = followers[i]
            rect = rectangles[i]
            if follower.is_ward:
                debug_draw_rectangles(
                    screen_copy, [rect], color=(0, 255, 0), width=5)
            else:
                debug_draw_rectangles(
                    screen_copy, [rect], color=(255, 0, 0),width=5)
        for i in range(len(follower_opponent)):
            follower = follower_opponent[i]
            rect = rectangles_opponent[i]
            if follower.is_ward:
                debug_draw_rectangles(
                    screen_copy, [rect], color=(0, 255, 0), width=5)
            else:
                debug_draw_rectangles(
                    screen_copy, [rect], color=(255, 0, 0), width=5)
        return screen_copy


class FeatureMatcher:
    def __init__(self, rects=None):
        """
        初始化特征匹配器
        :param rects: 外部提供的ROI区域配置字典 {模板名: (x1,y1,x2,y2)}
        """
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.match_ratio = 0.7
        self.min_matches = 10
        self.rects = rects or {}  # 外部ROI配置
        self.template_name = None  # 当前使用的模板名

    def set_template(self, name):
        """设置当前模板名用于自动获取ROI"""
        self.template_name = name

    def _get_roi(self, img, roi_spec):
        """根据ROI规范裁剪图像"""
        if roi_spec is None:
            return img

        if isinstance(img, Image.Image):
            w, h = img.size
            img = np.array(img)
        else:
            h, w = img.shape[:2]

        # 检查是否是相对坐标 (0-1之间的float)
        is_relative = (
                len(roi_spec) == 4 and
                all(isinstance(x, float) for x in roi_spec) and
                all(0 <= x <= 1 for x in roi_spec))

        # 支持绝对坐标和相对坐标
        if is_relative:
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(roi_spec, [w, h, w, h])]
        else:
            x1, y1, x2, y2 = roi_spec

        return img[y1:y2, x1:x2]

    def match_features(self, img1, img2, draw_matches=False):
        """
        特征匹配核心方法
        :param img1: 图像1 (numpy array)
        :param img2: 图像2 (numpy array)
        :param draw_matches: 是否返回匹配可视化图像
        :return: 匹配点数量 或 (匹配点数量, 可视化图像)
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return (0, None) if draw_matches else 0

        try:
            matches = self.flann.knnMatch(des1, des2, k=2)
        except cv2.error:
            return (0, None) if draw_matches else 0

        good_matches = [m for m, n in matches if m.distance < self.match_ratio * n.distance]

        if draw_matches:
            vis_img = cv2.drawMatches(
                img1, kp1, img2, kp2, good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            return len(good_matches), vis_img
        return len(good_matches)

    def find_best_match(self, query_img, templates_dict, roi=None, show_result=False):
        """
        增强版特征匹配
        :param query_img: 查询图像 (PIL/numpy array)
        :param templates_dict: {模板名: 模板图像}
        :param roi: 可覆盖模板预设ROI (x1,y1,x2,y2)
        :param show_result: 是否返回可视化结果
        :return: 匹配结果
        """
        # 自动获取预设ROI
        if roi is None and self.template_name in self.rects:
            roi = self.rects[self.template_name]

        # 处理查询图像
        query_np = np.array(query_img) if isinstance(query_img, Image.Image) else query_img
        query_roi = self._get_roi(query_np, roi)

        best_match = None
        best_count = 0
        best_vis = None

        for name, template in templates_dict.items():
            template_np = np.array(template) if isinstance(template, Image.Image) else template

            if show_result:
                count, vis = self.match_features(template_np, query_roi, draw_matches=True)
                if count > best_count:
                    best_count = count
                    best_match = name
                    best_vis = vis
            else:
                count = self.match_features(template_np, query_roi)
                if count > best_count:
                    best_count = count
                    best_match = name

        if show_result:
            return best_match, best_count, best_vis
        return best_match, best_count