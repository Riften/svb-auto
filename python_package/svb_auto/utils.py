from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np

def debug_draw_rectangles_relative(
        image: Image.Image,
        rectangles: list,
        color=(255, 0, 0),
        width=2) -> Image.Image:
    """在图像上绘制相对矩形框。
    :param image: 要绘制的图像。
    :param rectangles: 矩形列表，每个矩形是一个元组 (x1, y1, x2, y2)，为矩形顶点相对于画幅的坐标，取值范围为 [0, 1]。
    :param color: 矩形的颜色，默认为红色。
    :param width: 矩形边框的宽度，默认为 2。
    :return: 绘制了矩形的图像。
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image object")
    img_w, img_h = image.size
    abs_rectangles = rectangles.copy()
    for i, rect in enumerate(rectangles):
        x1, y1, x2, y2 = rect
        abs_rectangles[i] = (int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h))
    return debug_draw_rectangles(image, abs_rectangles, color, width)

def debug_draw_rectangles(
        image: Image.Image, 
        rectangles: list, 
        color=(255, 0, 0), 
        width=2) -> Image.Image:
    """
    在图像上绘制矩形框。

    :param image: 要绘制的图像。
    :param rectangles: 矩形列表，每个矩形是一个元组 (x1, y1, x2, y2)。
    :param color: 矩形的颜色，默认为红色。
    :param width: 矩形边框的宽度，默认为 2。
    :return: 绘制了矩形的图像。
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image object")

    draw = ImageDraw.Draw(image)
    for rect in rectangles:
        draw.rectangle(rect, outline=color, width=width)
    
    return image

def debug_draw_points(
        image: Image.Image,
        points: list,
        color=(255, 0, 0),
        radius=5) -> Image.Image:
    """
    在图像上绘制点。

    :param image: 要绘制的图像。
    :param points: 点列表，每个点是一个元组 (x, y)。
    :param color: 点的颜色，默认为红色。
    :param radius: 点的半径，默认为 5。
    :return: 绘制了点的图像。
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image object")

    draw = ImageDraw.Draw(image)
    for point in points:
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    
    return image

def debug_draw_points_relative(
        image: Image.Image,
        points: list,
        color=(255, 0, 0),
        radius=5) -> Image.Image:
    """
    在图像上绘制相对坐标的点。
    :param image: 要绘制的图像。
    :param points: 点列表，每个点是一个元组 (x, y)，取值范围为 [0, 1]。
    :param color: 点的颜色，默认为红色。
    :param radius: 点的半径，默认为 5。
    :return: 绘制了点的图像。
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image object")
    img_w, img_h = image.size
    abs_points = [(int(x * img_w), int(y * img_h)) for x, y in points]
    return debug_draw_points(image, abs_points, color, radius)


def crop_rectangle(
        image: Image.Image,
        rectangle: tuple
):
    """
    裁剪图像中的指定矩形区域。

    :param image: 要裁剪的图像。
    :param rectangle: 矩形坐标 (x1, y1, x2, y2)。
    :return: 裁剪后的图像。
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image object")
    x1, y1, x2, y2 = rectangle
    return image.crop((x1, y1, x2, y2))

def crop_rectangle_relative(
    image: Image.Image,
    rectangle: tuple
):
    """
    根据相对坐标裁剪图像中的指定矩形区域。

    :param image: 要裁剪的图像。
    :param rectangle: 相对矩形坐标 (x1, y1, x2, y2)，范围为[0, 1]。
    :return: 裁剪后的图像。
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image object")
    img_w, img_h = image.size
    x1, y1, x2, y2 = rectangle
    abs_rect = (int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h))
    return image.crop(abs_rect)

def get_rectangle_center_relative(
    rectangle: tuple,
    image_width: int,
    image_height: int,
    area_width: int = None,
    area_height: int = None,
    detected_position: tuple = None
) -> tuple:
    """
    获取矩形的中心点在屏幕坐标

    :param rectangle: 相对矩形坐标 (x1, y1, x2, y2)，范围为[0, 1]。
    :return: 中心点坐标 (x, y)，范围为[0, 1]。
    """
    x1, y1, x2, y2 = rectangle
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (int(center_x * image_width), int(center_y * image_height))