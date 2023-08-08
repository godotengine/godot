#!/bin/bash -e

VERSION=0.10.0

rm -rf AUTHORS LICENSE inc/ src/ *.zip *.tar.gz tmp/

mkdir tmp/ && pushd tmp/

curl -L -O https://github.com/thorvg/thorvg/archive/v$VERSION.tar.gz
tar --strip-components=1 -xvf *.tar.gz
rm *.tar.gz
find . -type f -name 'meson.build' -delete

# Fix newline at end of file.
for source in $(find ./ -type f \( -iname \*.h -o -iname \*.cpp \)); do
    sed -i -e '$a\' $source
done

cp -v AUTHORS LICENSE ..
cp -rv inc ../

cat << EOF > ../inc/config.h
#ifndef THORVG_CONFIG_H
#define THORVG_CONFIG_H

#define THORVG_SW_RASTER_SUPPORT

#define THORVG_SVG_LOADER_SUPPORT

#define THORVG_VERSION_STRING "$VERSION"
#endif
EOF

mkdir ../src
cp -rv src/lib ../src/
# Only sw_engine is enabled.
rm -rfv ../src/lib/gl_engine

# Only svg loader is enabled.
mkdir ../src/loaders
cp -rv src/loaders/svg src/loaders/raw  ../src/loaders/

# Future versions
# cp -rv src/utils ../src

popd
rm -rf tmp/
