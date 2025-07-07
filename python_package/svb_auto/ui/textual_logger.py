from textual.widgets import Log
from datetime import datetime


class WidgetLogger(Log):
    """
    Implement python Logger interface for Textual Log widget.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def info(self, message):
        """
        Log an info message.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[info][{timestamp}] {message}[/]"
        self.write(message)
        
    def error(self, message):
        """ 
        Log an error message.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[error][{timestamp}] {message}[/]"
        self.write(message)

    def debug(self, message):
        """
        Log a debug message.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[debug][{timestamp}] {message}[/]"
        self.write(message)

    def warning(self, message):
        """
        Log a warning message.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[warning][{timestamp}] {message}[/]"
        self.write(message)