#!/bin/bash

$SHELL -c "env -i scons platform=linux target=template_release arch=x86_64 $1"
$SHELL -c "env -i scons platform=linux target=template_debug arch=x86_64 $1"
$SHELL -c "env -i scons platform=linux target=template_release arch=x86_32 $1"
$SHELL -c "env -i scons platform=linux target=template_debug arch=x86_32 $1"
