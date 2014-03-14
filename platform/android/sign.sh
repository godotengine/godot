#!/bin/bash

jarsigner -digestalg SHA1 -sigalg MD5withRSA -verbose -keystore /home/luis/Downloads/carnavalguachin.keystore -storepass 12345678 "$1" momoselacome

echo ""
echo ""
echo "Checking if APK is verified..."
jarsigner -verify "$1" -verbose -certs

