#!/bin/bash

echo
echo "Download and install ccache ..."
echo

echo "Downloading sources ..."
curl -L -O https://www.samba.org/ftp/ccache/ccache-3.3.4.tar.gz # latest version available here: https://ccache.samba.org/download.html

echo "Extracting to build directory ..."
tar xzf ccache-3.3.4.tar.gz
cd ccache-3.3.4

echo "Compiling sources ..."
./configure --prefix=/usr/local --with-bundled-zlib > /dev/null
make

echo "Installing ..."

mkdir /usr/local/opt/ccache

mkdir /usr/local/opt/ccache/bin
cp ccache /usr/local/opt/ccache/bin
ln -s /usr/local/opt/ccache/bin/ccache /usr/local/bin/ccache

mkdir /usr/local/opt/ccache/libexec
links=(
  clang
  clang++
  cc
  gcc gcc2 gcc3 gcc-3.3 gcc-4.0 gcc-4.2 gcc-4.3 gcc-4.4 gcc-4.5 gcc-4.6 gcc-4.7 gcc-4.8 gcc-4.9 gcc-5 gcc-6 gcc-7
  c++ c++3 c++-3.3 c++-4.0 c++-4.2 c++-4.3 c++-4.4 c++-4.5 c++-4.6 c++-4.7 c++-4.8 c++-4.9 c++-5 c++-6 c++-7
  g++ g++2 g++3 g++-3.3 g++-4.0 g++-4.2 g++-4.3 g++-4.4 g++-4.5 g++-4.6 g++-4.7 g++-4.8 g++-4.9 g++-5 g++-6 g++-7
)
for link in "${links[@]}"; do
  ln -s ../bin/ccache /usr/local/opt/ccache/libexec/$link
done
#/usr/local/bin/ccache -M 2G
cd $TRAVIS_BUILD_DIR

echo
echo "Done!"
echo
