#!/bin/bash -e

VERSION=0.12.7

cd thirdparty/thorvg/ || true
rm -rf AUTHORS LICENSE inc/ src/ *.zip *.tar.gz tmp/

mkdir tmp/ && pushd tmp/

# Release
curl -L -O https://github.com/thorvg/thorvg/archive/v$VERSION.tar.gz
# Current Github main branch tip
#curl -L -O https://github.com/thorvg/thorvg/archive/refs/heads/main.tar.gz

tar --strip-components=1 -xvf *.tar.gz
rm *.tar.gz

# Install from local git checkout "thorvg-git" in the same directory
# as godot git checkout.
#d="../../../../thorvg-git"
#cp -r ${d}/AUTHORS ${d}/inc ${d}/LICENSE ${d}/src .

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
#define THORVG_PNG_LOADER_SUPPORT
#define THORVG_JPG_LOADER_SUPPORT
#define THORVG_THREAD_SUPPORT

// For internal debugging:
//#define THORVG_LOG_ENABLED

#define THORVG_VERSION_STRING "$VERSION"
#endif
EOF

mkdir ../src
cp -rv src/common ../src
cp -rv src/renderer ../src/

# Only sw_engine is enabled.
rm -rfv ../src/renderer/gl_engine
rm -rfv ../src/renderer/wg_engine

# Enabled embedded loaders: raw, JPEG, PNG.
mkdir ../src/loaders
cp -rv src/loaders/svg src/loaders/raw  ../src/loaders/
cp -rv src/loaders/jpg  ../src/loaders/
cp -rv src/loaders/png src/loaders/external_png  ../src/loaders/

popd
rm -rf tmp

