from abc import ABC, abstractmethod


class BaseClient(ABC):
    """
    Common interface for all uk-agents clients.

    Every client (REST, cron, Telegram, Streamlit) must implement:
      - start() — blocks until the client stops (serve, poll, loop, ...)
      - stop()  — signals a graceful shutdown; default is a no-op

    The class also acts as a context manager so clients can be used with
    `with RestClient() as client: client.start()`.
    """

    @abstractmethod
    def start(self) -> None:
        """Start the client and block until it stops."""

    def stop(self) -> None:
        """Signal a graceful shutdown. Override when cleanup is required."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
