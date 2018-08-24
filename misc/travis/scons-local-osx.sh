#!/bin/bash

echo
echo "Download and install Scons local package ..."
echo

echo "Downloading sources ..."
curl -L -O https://downloads.sourceforge.net/scons/scons-local-3.0.1.zip # latest version available here: http://scons.org/pages/download.html

echo "Extracting to build directory ..."
unzip -qq -n scons-local-3.0.1.zip -d $TRAVIS_BUILD_DIR/scons-local

echo "Installing symlinks ..."
ln -s $TRAVIS_BUILD_DIR/scons-local/scons.py /usr/local/bin/scons

echo
echo "Done!"
echo
