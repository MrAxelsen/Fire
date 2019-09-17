
import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65433        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	s.connect((HOST, PORT))
	s.sendall('request_angle'.encode())
	data = s.recv(1024).decode()
	print(data)
	s.sendall('shutdown'.encode())

print('Received', repr(data))
