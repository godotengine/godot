#!/usr/bin/env bash

# Generate .ico, .icns and .zip set of icons for Steam

# Make icons with transparent backgrounds and all sizes
for s in 16 24 32 48 64 128 256 512 1024; do
  convert -resize ${s}x$s -antialias \
          -background transparent \
          ../../icon.svg icon$s.png
done

# 16px tga file for library
convert icon16.png icon16.tga

# zip for Linux
zip godot-icons.zip icon*.png

# ico for Windows
# Not including biggest ones or it blows up in size
icotool -c -o godot-icon.ico icon{16,24,32,48,64,128,256}.png

# icns for macOS
# Only some sizes: http://iconhandbook.co.uk/reference/chart/osx/
png2icns godot-icon.icns icon{16,32,128,256,512,1024}.png

rm -f icon*.png
