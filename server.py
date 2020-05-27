import sys
from socket import socket, AF_INET, SOCK_DGRAM
import time


class DopeServer:
    def __init__(self, SERVER_IP='127.0.0.1', PORT_NUMBER=54000, SIZE=256):
        self.SERVER_IP = SERVER_IP
        self.PORT_NUMBER = PORT_NUMBER
        self.SIZE = SIZE

        self.mySocket = socket(AF_INET, SOCK_DGRAM)
        self.mySocket.connect((SERVER_IP, PORT_NUMBER))
        print("Test client sending packets to IP {0}, via port {1}\n".format(SERVER_IP, PORT_NUMBER))

    def list2str(self, inList):
        outStr = ",".join(str(e) for e in inList)
        return outStr

    # def list2str(self, inList):
    #     outStr = ""
    #     for index, item in enumerate(inList):
    #         outStr += str(item)
    #         if index !=len(inList)-1:
    #             outStr += ','
    #     return outStr

    def send(self, message):
        message = self.list2str(message)
        self.mySocket.send(message.encode('utf-8'))
        print("Sending: " + message)

# i = 0
# my_server = DopeServer()
#
# while True:
#     message = [i, i + 1, i + 2, 2*i, 2*i, 2*i, 2*i]
#     my_server.send(message)
#     i += 1
#     time.sleep(.5)
