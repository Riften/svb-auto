from textual.app import App
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Log


class ContainerLog(Container):
    DEFAULT_CSS = """
    ContainerLog {
        width: 1fr;
        height: 1fr;
        display: none;
        layer: above;
        margin: 3 3;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Log Output"
        self.id = "debug_log_container"

    def compose(self):
        yield Log(id="debug_log_widget")

class ControlPanel(Vertical):
    DEFAULT_CSS = """
    ControlPanel {
        border: solid round $primary;
        height: 100%;
        width: 20%;
        layer: below;
    }
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Control Panel"

class SVBAutoApp(App):
    """Main application class for SVB Auto."""

    DEFAULT_CSS = """
    SVBAutoApp {
        layers: below above;
    }
    """

    def __init___(self):
        super().__init__()
        self.title = "SVB Auto Application"
    
    def compose(self):
        """Compose the main layout of the application."""
        # Here you would define your application's layout
        with Horizontal():
            yield ControlPanel()
        yield ContainerLog()
                        

if __name__ == "__main__":
    app = SVBAutoApp()
    app.run()