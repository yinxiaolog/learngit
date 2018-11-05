# coding=utf-8
__author__ = 'yinxiaolog'
'''
server端
长连接，短连接，心跳
'''
import socket

def server():
    BUF_SIZE = 1024
    host = '192.168.50.159'
    port = 8443
    log = open('log.txt','a')

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)  # 接收的连接数
    client, address = server.accept()  # 因为设置了接收连接数为1，所以不需要放在循环中接收
    while True:  # 循环收发数据包，长连接
        data = client.recv(BUF_SIZE)
        log.write("hello world")
        print(data.decode())  # python3 要使用decode
        # client.close() #连接不断开，长连接

def main():
    server()

if __name__=='__main__':
    main()