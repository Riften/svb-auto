from re import A
from pytest import fail
from svb_auto.connect import connect_with_adbutils
from uiautomator2 import Device
from PIL import Image
import numpy as np
from svb_auto.utils import debug_draw_rectangles_relative
from svb_auto.detect import detect_template_in_area, Detector
from svb_auto.utils import crop_rectangle_relative
import cv2
from enum import Enum, auto
import time
import argparse
from datetime import datetime

MAX_FAILURE_COUNT = 50 # 最大失败次数
class AppState(Enum):
    UNKNOWN = auto() # 未知状态
    EXITED = auto() # 应用已退出
    STARTING = auto() # 启动界面，需要点击屏幕中央
    MAIN = auto() # 应用在主界面
    BATTLE_SELECT = auto() # 在对战模式选择界面
    TREASURE_RESULT = auto() # 宝箱奖励结果界面，点击跳过
    NETWORK_ISSUE = auto() # 网络不稳定弹出超时等对话框
    RETURN_TO_BATTLE = auto() # 返回对战窗口标题，闪退再打开会出现该窗口

    # 进入对战之后的状态
    BATTLE_DEFAULT = auto() # 默认对战状态
    BATTLE_OPPONENT_TURN = auto() # 对手回合
    BATTLE_PLAYER_TURN = auto() # 玩家回合
    BATTLE_SWAP_CARD = auto() # 换牌阶段
    

class MatchOperator(Enum):
    AND = auto() # 与操作
    OR = auto()  # 或操作

map_state_template = {
    AppState.EXITED: (["app_icon"], MatchOperator.OR),
    AppState.STARTING: (["starting"], MatchOperator.OR),
    AppState.MAIN: (["money", 'crystal'], MatchOperator.AND),
    AppState.BATTLE_SELECT: (["battle_start"], MatchOperator.OR),
    AppState.BATTLE_SWAP_CARD: (["decision"], MatchOperator.OR),
    AppState.BATTLE_PLAYER_TURN: (["end_round"], MatchOperator.OR),
    AppState.TREASURE_RESULT: (["treasure_result"], MatchOperator.OR),
    AppState.NETWORK_ISSUE: (['retry'], MatchOperator.OR),
    AppState.RETURN_TO_BATTLE: (['return_to_battle'], MatchOperator.OR),  # 返回对战窗口标题，闪退再打开会出现该窗口
}

map_battle_state_template = {
    AppState.BATTLE_SWAP_CARD: (["decision"], MatchOperator.OR),
    AppState.BATTLE_PLAYER_TURN: (["end_round"], MatchOperator.OR),
    AppState.BATTLE_SELECT: (["battle_start"], MatchOperator.OR),
    AppState.TREASURE_RESULT: (["treasure_result"], MatchOperator.OR),
}

special_points = {
    'battle_1': (0.31, 0.90),
    'random_battle': (666/1920, 577/1080),  # 随机对战按钮位置
    'decision': (1729/1920, 545/1080),  # 对战期间决定按钮，包括换牌，结束回合，都点击这里
    'card_hand_small': (1484/1920, 1014/1080),  # 手牌小图标位置，用来点击放大手牌列表
    # 手牌大图标位置，用来使用手牌
    'card_hand_large': [(960/1920, 960/1080), 
                        (812/1920, 960/1080),
                        (1216/1920, 960/1080),
                        (618/1920, 960/1080),
                        (1439/1920, 960/1080),],
    'card_field': [(1020/1920, 620/1080), 
                    (1247/1920, 620/1080),
                    (1474/1920, 620/1080),
                    (793/1920, 620/1080),
                    (566/1920, 620/1080),],
    'opponent': (960/1920, 85/1080),  # 对手主站者位置
    'return_to_battle_negative': (769/1920, 838/1080),  # 返回对战窗口 “否” 按钮位置
    'return_to_battle_second': (960/1920, 838/1080),  # 返回对战第二个窗口 “确认” 按钮位置
}

class App:
    """
    App 是一个简单的状态机，用于根据应用状态执行不同的操作。总体上他的逻辑是
    1. 在应用退出时重启应用
    2. 在应用启动后进入阶位对战
    3. 进入对战逻辑
    4. 对战结束后继续匹配
    5. 任何时候出现连接问题等回到主页，则重新进入阶位对战
    6. 任何时候报错闪退，则重新进入应用

    其实现逻辑为，on_xxx 函数执行当前状态下的操作，不会检测执行是否成功。
    但如果达到最大尝试次数之后依然无法决定应该执行的操作（例如无法检测到图标），
    则会返回特定状态，让状态机重新检测当前所处状态。
    """
    def __init__(
            self, 
            port: int = 16384,
            img_dir: str = "imgs_chs_1920_1080",
            screen_interval: float = 1,
            skip_mode: bool = False,
            app_name = None):
        self.device = connect_with_adbutils(port)
        self.detector = Detector(img_dir=img_dir)
        self.screen_interval = screen_interval
        self.app_name = app_name

        self.unknown_state_start_time = None

        test_screenshot = self.device.screenshot()
        if not isinstance(test_screenshot, Image.Image):
            raise ValueError("无法获取设备屏幕截图，请检查设备连接是否正常。")
        self.image_width, self.image_height = test_screenshot.size
        self.map_handlers = {
            AppState.UNKNOWN: self.detect_state,
            AppState.EXITED: self.on_exited,
            AppState.STARTING: self.on_starting,
            AppState.MAIN: self.on_main,
            AppState.BATTLE_SELECT: self.on_battle_select,
            AppState.BATTLE_DEFAULT: self.detect_battle_state,
            AppState.BATTLE_SWAP_CARD: self.on_drop_card,
            AppState.BATTLE_PLAYER_TURN: self.on_player_turn,
            AppState.TREASURE_RESULT: self.on_treasure_result,
            AppState.NETWORK_ISSUE: self.on_retry,
        }
        if skip_mode:
            # 如果是跳过模式，则只处理对战状态
            self.map_handlers[AppState.BATTLE_PLAYER_TURN] = self.on_player_turn_skip

        # some status variables
        self.fail_count = 0  # 用于记录失败次数，部分当前没处理的状态可能导致死循环，此时会尝试点击屏幕中央，然后返回 UNKNOWN 状态

    def abs_position(self, relative_position: tuple[float, float]) -> tuple[int, int]:
        """
        将相对位置转换为绝对位置
        :param relative_position: 相对位置，范围 [0, 1]
        :return: 绝对位置
        """
        return (
            int(relative_position[0] * self.image_width),
            int(relative_position[1] * self.image_height)
        )

    def run(self):
        current_state = AppState.UNKNOWN
        while True:
            func = self.map_handlers.get(current_state, None)
            if func is None:
                print(f"\033[31m[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\033[0m 未知状态: {current_state}, 无法处理")
                break
            if current_state == AppState.UNKNOWN:
                if self.unknown_state_start_time is None:
                    self.unknown_state_start_time = datetime.now()
                else:
                    # 删除控制台的上三行
                    for _ in range(3):
                        print("\033[A\033[K", end="")
                elapsed_time = datetime.now() - self.unknown_state_start_time
                print(f"\033[31m[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\033[0m 未知状态: {current_state}, 已持续: {str(elapsed_time).split('.')[0]}")
            else:
                self.unknown_state_start_time = None
                print(f"\033[32m[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\033[0m 当前状态: {current_state}, 执行操作")
            try:
                current_state = func()
            except Exception as e:
                print(f"执行操作时发生错误: {e}")
                current_state = AppState.UNKNOWN
            
            time.sleep(self.screen_interval)  # 等待一段时间，避免过于频繁的操作
            if current_state == AppState.UNKNOWN or current_state == AppState.BATTLE_DEFAULT:
                self.fail_count += 1
                if self.fail_count >= MAX_FAILURE_COUNT:
                    print("连续失败次数过多，点击屏幕中央")
                    self.click_center()
                    current_state = AppState.UNKNOWN # 重置状态为 UNKNOWN
            else:
                self.fail_count = 0
    def click_center(self):
        """
        点击屏幕中央
        """
        print("点击屏幕中央")
        
        self.device.click(self.image_width // 2, self.image_height // 2)

    def click_relative(self, relative_position: tuple[float, float]):
        """
        点击相对位置
        :param relative_position: 相对位置，范围 [0, 1]
        """
        print(f"点击相对位置: {relative_position}")
        position = self.abs_position(relative_position)
        self.device.click(*position)

    def detect_state(self) -> AppState:
        """
        检测当前应用状态
        """
        print("检测当前应用状态")
        screen = self.device.screenshot()
        for state, (templates, operator) in map_state_template.items():
            res = []
            
            for template in templates:
                # print(f"检测状态: {state}, 模板: {template}")
                detected, value, position = self.detector.detect_on_screen(
                    screen,
                    template_name=template,
                    threshold=0.8
                )
                res.append(detected)
            if operator == MatchOperator.AND:
                is_detected = all(res)
            else:
                is_detected = any(res)
            if is_detected:
                print(f"检测到状态: {state}, 模板: {templates}, 置信度: {value}, 位置: {position}")
                return state
        print("未检测到任何已知状态，返回 UNKNOWN")
        return AppState.UNKNOWN
    
    def detect_battle_state(self) -> AppState:
        """
        检测当前对战状态
        """
        print("检测当前对战状态")
        
        screen = self.device.screenshot()
        for state, (templates, operator) in map_battle_state_template.items():
            res = []
            for template in templates:
                detected, value, position = self.detector.detect_on_screen(
                    screen,
                    template_name=template,
                    threshold=0.8
                )
                res.append(detected)
            if operator == MatchOperator.AND:
                is_detected = all(res)
            else:
                is_detected = any(res)
            if is_detected:
                print(f"检测到对战状态: {state}, 模板: {templates}, 置信度: {value}, 位置: {position}")
                return state
        
        print("未检测到任何已知对战状态，返回 BATTLE_DEFAULT")
        return AppState.BATTLE_DEFAULT
    
    def detect_and_click(self, 
            template_name: str,
            threshold: float = 0.8,
            click_position: tuple[float, float] = None,
            method=cv2.TM_CCOEFF_NORMED) -> bool:
        screen = self.device.screenshot()
        is_detected, value, position = self.detector.detect_on_screen(
            screen,
            template_name=template_name,
            threshold=threshold,
            method=method
        )
        if is_detected:
            print(f"检测到 {template_name}: {is_detected}, 置信度: {value}, 位置: {position}")
            if click_position is not None:
                # 如果指定了点击位置，则点击该位置
                self.device.click(*self.abs_position(click_position))
            else:
                # 否则点击检测到的位置
                self.device.click(*position)
        return is_detected

    def save_screenshot(self, filename: str = 'screenshot.png'):
        """
        保存当前屏幕截图
        """
        screen = self.device.screenshot()
        if isinstance(screen, Image.Image):
            screen.save(filename)
            print(f"屏幕截图已保存到 {filename}")
        else:
            print("无法保存屏幕截图，返回的不是图像对象")

    def on_exited(self):
        """
        应用已退出，重新启动应用
        """
        print("应用已退出，重新启动应用")
        fail_count = 0
        if self.app_name is None:
            while True:
                
                is_detected = self.detect_and_click("app_icon")
                if not is_detected:
                    fail_count += 1
                    if fail_count >= MAX_FAILURE_COUNT:
                        print("无法检测到应用图标，返回 UNKNOWN 状态")
                        return AppState.UNKNOWN
                else:
                    time.sleep(10)  # 等待应用启动
                    return AppState.STARTING
                time.sleep(1)
        else:
            self.device.app_start(self.app_name)
            self.device.wait_activity(f"{self.app_name}.MainActivity", timeout=10)
        return AppState.STARTING
    
    def on_starting(self):
        """
        应用在启动界面，点击屏幕中央
        """
        
        fail_count = 0
        while True:
            screen = self.device.screenshot()
            is_detected, value, position = self.detector.detect_on_screen(
                screen,
                template_name="starting",
                threshold=0.8
            )
            if is_detected:
                print(f"检测到启动界面: {is_detected}, 置信度: {value}, 位置: {position}")
                print("应用在启动界面，点击屏幕中央")
                self.click_center()
                fail_count += 1
                if fail_count >= MAX_FAILURE_COUNT:
                    print("无法处理启动界面，返回 UNKNOWN 状态")
                    return AppState.UNKNOWN
                time.sleep(3)
                # 启动界面可能排队，因此循环多次调用，直到无法再检测到启动界面的标志
            else:
                # 理论上说，之后会进入主界面。这里为了维持鲁棒性（例如随后处理首次启动的卡包的信息）
                # 返回 UNKNOWN 状态，重新检测状态
                return self.detect_state()
    def on_treasure_result(self):
        self.click_center()
        return AppState.UNKNOWN
    
    def on_main(self):
        """
        应用在主界面，执行主界面的操作
        """
        print("处理主界面")
        fail_count = 0
        while True:
            screen = self.device.screenshot()
            # 检测是否处于主页对战选项卡
            is_detected, value, position = self.detector.detect_on_screen(
                screen,
                template_name="private_battle",
                threshold=0.8
            )
            if is_detected:
                self.click_relative(special_points['random_battle'])
                return AppState.BATTLE_SELECT
            # 点击 special_points['battle_1']，尝试进入对战选项卡
            position = special_points['battle_1']
            self.device.click(
                int(position[0] * self.image_width),
                int(position[1] * self.image_height)
            )
            fail_count += 1
            if fail_count >= MAX_FAILURE_COUNT:
                print("无法处理主界面，返回 UNKNOWN 状态")
                return AppState.UNKNOWN
            time.sleep(1) # 等待对战界面加载

    def on_battle_select(self):
        """
        处理对战选择界面
        """
        print("处理对战选择界面")
        fail_count = 0
        while True:
            is_detected = self.detect_and_click('battle_start')

            if is_detected:
                print("检测到对战开始按钮，点击进入对战")
                # 点击对战开始按钮后，直接返回 UNKNOWN 状态，交由状态机判断是否进入了战斗
                return AppState.UNKNOWN
            else:
                fail_count += 1
                if fail_count >= MAX_FAILURE_COUNT:
                    print("无法处理对战选择界面，返回 UNKNOWN 状态")
                    # 如果无法检测到对战开始按钮，则点击屏幕中央，尝试重新进入对战选择界面
                    self.click_center()
                    return AppState.UNKNOWN
                
    
    def on_drop_card(self):
        """
        换牌阶段，点击决策按钮
        """
        print("处理换牌阶段")
        self.click_relative(special_points['decision'])
        return AppState.BATTLE_DEFAULT
    
    def on_player_turn_skip(self):
        """
        玩家回合，空过
        """
        print("处理玩家回合")
        print("点击结束回合按钮")
        self.click_relative(special_points['decision'])
        time.sleep(0.2)  # 等待对战状态更新

        self.detect_and_click(
            "end",
            threshold=0.8,
        )

        return AppState.BATTLE_DEFAULT

    def on_player_turn(self):
        """
        玩家回合，点击结束回合按钮
        """
        print("处理玩家回合")
        print("尝试使用手牌")
        # 点击手牌小图标位置，放大手牌列表
        
        for card_position in special_points['card_hand_large']:
            self.click_relative(special_points['card_hand_small'])
            time.sleep(0.2)
            # 从 card_position 拖拽到屏幕中心
            start_pos = self.abs_position(card_position)
            end_pos = (self.image_width // 2, self.image_height // 2)
            self.device.swipe(start_pos[0], start_pos[1], end_pos[0], end_pos[1], duration=0.1)

            # 部分法术卡牌可能使用后需要确认，或者需要选取对象，这里统一在使用卡牌后点击一下对方主站者，可以有效避免卡死
            time.sleep(0.1)
            self.click_relative(special_points['opponent'])
            time.sleep(0.2)
            
            
        
        print("尝试进化并攻击对方主站者")
        screen = self.device.screenshot()
        can_envolve, value, position = self.detector.detect_on_screen(
            screen,
            template_name="can_envolve",
        )
        can_super_envolve, value, position = self.detector.detect_on_screen(
            screen,
            template_name="can_super_envolve",
        )
        for field_position in special_points['card_field']:
            # 点击手牌大图标位置，使用手牌
            self.click_relative(field_position)
            time.sleep(0.1)

            if can_super_envolve:
                print("检测到可以超进化")
                is_detected = self.detect_and_click(
                    template_name="super_envolve",
                )
                if is_detected:
                    print("等待超进化")
                    time.sleep(5)
                    can_super_envolve = False # 一回合只进化一次
                    can_envolve = False 
            else:
                if can_envolve:
                    print("检测到可以进化")
                    is_detected = self.detect_and_click(
                        template_name="envolve",
                    )
                    if is_detected:
                        print("等待进化")
                        time.sleep(5)
                        can_super_envolve = False # 一回合只进化一次
                        can_envolve = False 

            start_pos = self.abs_position(field_position)
            end_pos = self.abs_position(special_points['opponent'])
            self.device.swipe(start_pos[0], start_pos[1], end_pos[0], end_pos[1], duration=0.1)
            time.sleep(0.2)
        print("点击结束回合按钮")
        self.click_relative(special_points['decision'])
        time.sleep(0.2)  # 等待对战状态更新
        # 检查是否弹出确认对话框
        self.detect_and_click(
            "end",
            threshold=0.8,
        )

        return AppState.BATTLE_DEFAULT

    def on_retry(self):
        """
        网络不稳定时点击重试
        """
        print("处理网络不稳定情况，点击重试")
        is_detected = self.detect_and_click(
            "retry",
            threshold=0.8,
        )
        return AppState.UNKNOWN
    
    def on_return_to_battle(self):
        """
        处理返回对战窗口的情况
        """
        print("处理返回对战窗口")
        # 点击否按钮
        self.click_relative(special_points['return_to_battle_negative'])
        time.sleep(1)
        # 点击确认按钮
        self.click_relative(special_points['return_to_battle_second'])
        return AppState.UNKNOWN
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVBYD 自动对战脚本")
    parser.add_argument("--port", type=int, default=16384, help="设备连接端口")
    parser.add_argument("--img_dir", type=str, default="imgs_chs_1920_1080", help="模板图片目录")
    parser.add_argument("--screen_interval", type=float, default=1.0, help="屏幕截图间隔时间")
    parser.add_argument("--app_name", type=str, default=None, help="应用包名（可选）")
    parser.add_argument("--skip_mode", action='store_true', help="是否启用空过模式，跳过玩家回合的操作")
    parser.add_argument("--server", type=bool, default=True, help="服务器: True国服, False国际服繁体")
    
    args = parser.parse_args()
    app = App(
        port=args.port,
        img_dir=args.img_dir + '/svwb' if args.server else args.img_dir + '/svwb_global',
        screen_interval=args.screen_interval,
        app_name=args.app_name,
        skip_mode=args.skip_mode
    )
    app.run()
