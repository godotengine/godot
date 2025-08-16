#!/usr/bin/env bash
dd if=$1 > /dev/tcp/$2/$3
