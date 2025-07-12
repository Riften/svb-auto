from textual.app import App
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Log, Header, Button, Input, Static, Select, Checkbox
from textual import work

from svb_auto.ui.textual_logger import WidgetLogger
from svb_auto.main import App as AUTOApp

# for release
import uiautomator2.assets

class AlertMsg(Static):
    DEFAULT_CSS = """
    AlertMsg {
        color: $accent;
        height: 5;
        text-align: center;
    }
    """

    CONTENT = """
    本项目完全开源且免费\n
    项目地址：[url]https://github.com/Riften/svb-auto[/]
    """

    def __init__(self, **kwargs):
        super().__init__(content=AlertMsg.CONTENT, markup=True, **kwargs)

class ContainerLog(Container):
    DEFAULT_CSS = """
    ContainerLog {
        border: solid round $primary;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Log Output"
        self.id = "debug_log_container"
        self.logger = WidgetLogger(id="log_debug")
    def compose(self):
        yield self.logger

class ControlPanel(Vertical):
    DEFAULT_CSS = """
    ControlPanel {
        border: solid round $primary;
    }
    ControlPanel > Button {
        height: 3;
        border: none;
    }
    ControlPanel > Input {
        height: 3;
        margin: 0 1;
    }
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Control Panel"

    def compose(self):
        yield Button("[blink][b]启动[/][/][green]▶[/]", id="button_start")
    
        yield Static("端口（MUMU 16384，雷模拟器 5555）")
        yield Input(placeholder="port number", id="input_port", value="16384", type="number")
        yield Static("服务器")
        yield Select(options=[("国服（简中）", 0), ("国际服（繁中）", 1)], value=0, id="select_server")
        yield Checkbox("完全空过模式", id="checkbox_skipmode", value=False)
        yield Checkbox("根据分组自动空过", id="checkbox_autoskip", value=True)

class SVBAutoApp(App):
    """Main application class for SVB Auto."""

    DEFAULT_CSS = """
    SVBAutoApp {
        
    }

    #main_container 
    {
        layout: grid;
        grid-size: 2 1;
        grid-rows: 1fr;
        grid-columns: 1fr;
        grid-gutter: 1;
    }
    """

    def __init___(self):
        super().__init__()
        self.title = "SVB Auto Application"
    
    def on_mount(self):
        self.theme = "dracula"

    def compose(self):
        """Compose the main layout of the application."""
        # Here you would define your application's layout
        yield Header()
        with Container(id="main_container"):
            with Vertical():
                yield AlertMsg()
                yield ControlPanel()
            yield ContainerLog()
    
    def get_input_value(self, id: str) -> str:
        widget = self.query_one(f"#{id}", Input)
        value = widget.value
        if not widget.validate(value):
            self.logger.error(f"Invalid input format for {id}")
        return value
    
    @work(exclusive=True, thread=True)
    def run_app(self):
        """Run the main application logic in a separate thread."""
        self.logger.info("Starting SVB Auto Application...")
        # Here you can add the logic to start your application
        # For example, initializing the app with the provided port and server
        port = self.get_input_value('input_port')
        server = self.query_one("#select_server", Select).value
        skip_mode = self.query_one("#checkbox_skipmode", Checkbox).value
        auto_skip = self.query_one("#checkbox_autoskip", Checkbox).value
        
        auto_app = AUTOApp(
            port=port, 
            img_dir="imgs_chs_1920_1080/svwb" if server == 0 else "imgs_int_1920_1080/svwb_global",
            logger=self.logger,
            skip_mode=skip_mode,
            enable_auto_skip=auto_skip
        )
        self.logger.info(f"Running SVB Auto with port {port}, server {server}, skip_mode {skip_mode}, auto_skip {auto_skip}")
        auto_app.run()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        
        if button_id == "button_start":
            self.logger.info("Start button pressed, running app...")
            event.button.disabled = True
            event.button.label = "[blink][b]正在运行...[/][/]"
            self.run_app()

    @property
    def logger(self) -> WidgetLogger:
        """Get the logger widget."""
        return self.query_one("#log_debug", WidgetLogger)

if __name__ == "__main__":
    app = SVBAutoApp()
    app.run()
    