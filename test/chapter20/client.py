# coding=utf-8
__author__ = 'yinxiaolog'
'''
client端
长连接，短连接，心跳
'''
import socket
import time
import sys

def client():
    host = '192.168.216.186'
    port = 8443
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # 在客户端开启心跳维护
    log = open('log.txt','a')
    
    try:
        client.connect((host, port))
    except:
        sys.exit(1)
    while True:
        client.send('hello world\r\n'.encode())
        log.write("send data")
        print('send data')
        time.sleep(1)  # 如果想验证长时间没发数据，SOCKET连接会不会断开，则可以设置时间长一点

def main():
    client()

if __name__=='__main__':
    main()