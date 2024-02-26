#!/bin/bash
scons platform=windows target=template_release arch=x86_32
scons platform=windows target=template_debug arch=x86_32


scons platform=windows target=template_release arch=x86_64
scons platform=windows target=template_debug arch=x86_64
