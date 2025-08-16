#!/bin/bash
varnishd -a :8080 -f $PWD/cache.vcl -F
