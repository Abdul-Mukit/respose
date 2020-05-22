import sys
from socket import socket, AF_INET, SOCK_DGRAM
import time

SERVER_IP = '127.0.0.1'
PORT_NUMBER = 54000
SIZE = 256
print("Test client sending packets to IP {0}, via port {1}\n".format(SERVER_IP, PORT_NUMBER))
mySocket = socket(AF_INET, SOCK_DGRAM)
mySocket.connect((SERVER_IP,PORT_NUMBER))

def list2str(inList):
    outStr = ",".join(str(e) for e in inList)
    return outStr

i = 0
while True:
    message = [i, i + 1, i + 2, 2*i, 2*i, 2*i, 2*i]
    message = list2str(message)
    # message = ",".join(str(e) for e in message)

    # message = input('-> ')
    mySocket.send(message.encode('utf-8'))
    print("Sending: " + message)
    i += 1
    time.sleep(.5)