#!/bin/bash
counter=$(ps -C httpd --no-heading|wc -l)
if [ "${counter}" = "0" ]; then
   ######systemctl start httpd
    sleep 2
    counter=$(ps -C httpd --no-heading|wc -l)
    if [ "${counter}" = "0" ]; then
        systemctl stop keepalived
    fi
fi
