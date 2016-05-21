#!/bin/sh
! egrep -r "\s+$" --include=*\.cpp --include=*\.h --exclude=./drivers/* .  > /dev/null
