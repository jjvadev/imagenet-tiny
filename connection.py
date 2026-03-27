from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5001
DEFAULT_BACKLOG = 8
HEADER_SIZE = 8


def recvall(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("La conexion se cerro mientras se recibian datos")
        data.extend(packet)
    return bytes(data)


def send_msg(sock: socket.socket, msg: Any) -> None:
    payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!Q", len(payload))
    sock.sendall(header + payload)


def recv_msg(sock: socket.socket) -> Any:
    header = recvall(sock, HEADER_SIZE)
    (length,) = struct.unpack("!Q", header)
    payload = recvall(sock, length)
    return pickle.loads(payload)


def create_server_socket(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    backlog: int = DEFAULT_BACKLOG,
) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(backlog)
    return server


def create_client_socket(host: str, port: int, timeout: float = 30.0) -> socket.socket:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(timeout)
    client.connect((host, port))
    return client
