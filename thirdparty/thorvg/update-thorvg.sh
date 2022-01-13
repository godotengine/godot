VERSION=0.7.0
rm -rf AUTHORS inc LICENSE src *.zip
curl -L -O https://github.com/Samsung/thorvg/archive/refs/tags/v$VERSION.zip
bsdtar --strip-components=1 -xvf *.zip
rm *.zip
rm -rf .github docs pc res test tools .git* *.md *.txt wasm_build.sh
find . -type f -name 'meson.build' -delete
rm -rf src/bin src/bindings src/examples src/wasm
rm -rf src/lib/gl_engine tvgcompat
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
