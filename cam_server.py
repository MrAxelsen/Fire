import socket

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65433        # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	s.bind((HOST, PORT))
	s.listen()
	conn, addr = s.accept()
	with conn:
		print('Connected by', addr)
		while True:
			data = c
			onn.recv(1024).decode()
			if data == 'request_angle':
				answer = '-17:22'
				print(str(data))
			elif data == 'shutdown':
				break
			conn.sendall(answer.encode())
