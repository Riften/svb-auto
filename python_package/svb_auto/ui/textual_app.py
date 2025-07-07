from venv import logger
from textual.app import App
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Log, Header, Button

from .textual_logger import WidgetLogger
import threading

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
        self.logger = WidgetLogger(id="debug_log_widget")
    def compose(self):
        yield self.logger

class ControlPanel(Vertical):
    DEFAULT_CSS = """
    ControlPanel {
        border: solid round $primary;
    }
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Control Panel"

    def compose(self):
        yield Button("Start")

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
            yield ControlPanel()
            yield ContainerLog()
    
    @property
    def logger(self) -> WidgetLogger:
        """Get the logger widget."""
        return self.query_one("#debug_log_widget", WidgetLogger)

if __name__ == "__main__":
    app = SVBAutoApp()
    app.run()
    