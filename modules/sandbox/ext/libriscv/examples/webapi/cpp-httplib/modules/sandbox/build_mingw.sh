CC=x86_64-w64-mingw32-gcc-win32
CXX=x86_64-w64-mingw32-g++-win32

mkdir -p .build_mingw
pushd .build_mingw
cmake .. -DCMAKE_BUILD_TYPE=Release -DRISCV_LIBTCC=ON -DMINGW_TOOLCHAIN=ON -DGODOT_DISABLE_EXCEPTIONS=OFF -DGODOTCPP_DISABLE_EXCEPTIONS=OFF -DGODOTCPP_TARGET=template_release
make -j16
popd

mv .build_mingw/libgodot-riscv.so .build_mingw/libgodot_riscv.windows.template_release.x86_64.dll
cp .build_mingw/*.dll /srv/samba/share
cp .build_mingw/libgodot_riscv.windows.template_release.x86_64.dll ~/New\ Game\ Project/extensions/bin/libgodot_riscv.windows.template_release.x86_64.dll
