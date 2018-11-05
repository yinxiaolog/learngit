#!/usr/bin/env bash
python server.py
python client.py

while true
do
    if [$?==0]
    then
        echo "连接成功"
        break
    else
        echo "正在连接"
        python client.py
    fi
done