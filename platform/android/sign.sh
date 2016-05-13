#!/bin/bash

jarsigner -digestalg SHA1 -sigalg MD5withRSA -verbose -keystore my-release-key.keystore "$1" reduz

echo ""
echo ""
echo "Checking if APK is verified..."
jarsigner -verify "$1" -verbose -certs

