#!/bin/bash

scons platform=linux target=template_release arch=x86_64
scons platform=linux target=template_debug arch=x86_64

scons platform=linux target=template_release arch=x86_32
scons platform=linux target=template_debug arch=x86_32
