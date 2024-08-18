#!/bin/bash

$SHELL -c "env -i scons platform=android target=template_release generate_bindings=yes ANDROID_HOME=/home/bardia/Android/Sdk $1"
$SHELL -c "env -i scons platform=android target=template_debug generate_bindings=yes ANDROID_HOME=/home/bardia/Android/Sdk $1"
