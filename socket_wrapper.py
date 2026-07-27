import socket
import logging

logger = logging.getLogger(__name__)


class SocketWrapper:
    def __init__(self, host, port):
        """
        Initialize a UDP socket wrapper for communication with Unity.

        Args:
            host (str): The IP address of the Unity application
            port (int): The port number of the Unity application
        """
        self.host = host
        self.port = port

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logger.info(f"UDP socket initialized for {self.host}:{self.port}")
        except socket.error as e:
            logger.error(f"Failed to initialize socket: {e}")
            raise

    def send(self, data):
        """
        Send data to the configured host and port via UDP.

        Args:
            data: The data to send (will be converted to string and encoded)
        """
        try:
            if not isinstance(data, str):
                data = str(data)

            self.sock.sendto(data.encode(), (self.host, self.port))
            logger.debug(f"Sent data: {data}")

        except socket.error as e:
            logger.error(f"Failed to send data: {e}")

    def close(self):
        """Close the socket connection."""
        try:
            self.sock.close()
            logger.info("Socket closed successfully")
        except socket.error as e:
            logger.error(f"Error closing socket: {e}")

    def __del__(self):
        """Ensure socket is closed when object is destroyed."""
        self.close()
