#!/bin/bash

$SHELL -c "env -i scons platform=windows target=template_release arch=x86_32 generate_bindings=yes $1"
$SHELL -c "env -i scons platform=windows target=template_debug arch=x86_32 generate_bindings=yes $1"
$SHELL -c "env -i scons platform=windows target=template_release arch=x86_64 generate_bindings=yes $1"
$SHELL -c "env -i scons platform=windows target=template_debug arch=x86_64 generate_bindings=yes $1"
