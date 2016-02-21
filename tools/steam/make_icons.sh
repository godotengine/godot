convert -resize 32x32 ../../godot_icon.svg icon32.ico
convert -resize 32x32 ../../godot_icon.svg icon32.icns
for s in 16 24 32 64 96 128 256; do convert -resize ${s}x$s ../../godot_icon.svg icon$s.png; done
zip icons.zip icon*.png
rm icon*.png


