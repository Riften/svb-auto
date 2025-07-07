from ast import In
from venv import logger
from textual.app import App
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Log, Header, Button, Input, Static

from .textual_logger import WidgetLogger
import threading

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
    
        yield Static("端口:")
        yield Input(placeholder="port number", id="input_port", value="16384", type="number")
        yield Static("server")


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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        start_args = {
            "port": -1,
        }
        if button_id == "button_start":
            port = self.get_input_value('input_port')
            start_args['port'] = int(port)
            self.logger.info(start_args)
            self.logger.info("SZB，启动！")

    @property
    def logger(self) -> WidgetLogger:
        """Get the logger widget."""
        return self.query_one("#log_debug", WidgetLogger)

if __name__ == "__main__":
    app = SVBAutoApp()
    app.run()
    