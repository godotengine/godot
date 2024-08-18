#!/bin/bash

scons platform=macos target=template_release $1
scons platform=macos target=template_debug $1
scons platform=ios target=template_release $1
scons platform=ios target=template_debug $1
