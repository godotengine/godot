VERSION=0.8.99
rm -rf AUTHORS inc LICENSE src

d="../../../thorvg-git"

cp -r ${d}/AUTHORS ${d}/inc ${d}/LICENSE ${d}/src .
find . -type f -name 'meson.build' -delete
rm -rf src/bin src/bindings src/examples src/wasm
rm -rf src/lib/gl_engine src/loaders/external_jpg src/loaders/png

cat << EOF > inc/config.h
#ifndef THORVG_CONFIG_H
#define THORVG_CONFIG_H

#define THORVG_SW_RASTER_SUPPORT 1

#define THORVG_SVG_LOADER_SUPPORT 1

#define THORVG_PNG_LOADER_SUPPORT 1

#define THORVG_TVG_LOADER_SUPPORT 1

#define THORVG_TVG_SAVER_SUPPORT 1

#define THORVG_JPG_LOADER_SUPPORT 1

#define THORVG_VERSION_STRING "$VERSION"
#endif
EOF
for source in $(find ./ -type f \( -iname \*.h -o -iname \*.cpp \)); do
    sed -i -e '$a\' $source
done
