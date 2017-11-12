#!/bin/bash

# SDK
# https://dl.google.com/android/repository/sdk-tools-linux-3859397.zip
# SHA-256 444e22ce8ca0f67353bda4b85175ed3731cae3ffa695ca18119cbacef1c1bea0
# latest version available here: https://developer.android.com/studio/index.html

# NDK
# https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip
# SHA-1 0bf02d4e8b85fd770fd7b9b2cdec57f9441f27a2
# latest version available here: https://developer.android.com/ndk/downloads/index.html

BASH_RC=~/.bashrc
GODOT_BUILD_TOOLS_PATH=./godot-dev/build-tools
mkdir -p $GODOT_BUILD_TOOLS_PATH
cd $GODOT_BUILD_TOOLS_PATH

ANDROID_BASE_URL=http://dl.google.com/android/repository

ANDROID_SDK_RELEASE=3859397
ANDROID_SDK_DIR=android-sdk
ANDROID_SDK_FILENAME=sdk-tools-linux-$ANDROID_SDK_RELEASE.zip
ANDROID_SDK_URL=$ANDROID_BASE_URL/$ANDROID_SDK_FILENAME
ANDROID_SDK_PATH=$GODOT_BUILD_TOOLS_PATH/$ANDROID_SDK_DIR
ANDROID_SDK_SHA256=444e22ce8ca0f67353bda4b85175ed3731cae3ffa695ca18119cbacef1c1bea0

ANDROID_NDK_RELEASE=r15c
ANDROID_NDK_DIR=android-ndk
ANDROID_NDK_FILENAME=android-ndk-$ANDROID_NDK_RELEASE-linux-x86_64.zip
ANDROID_NDK_URL=$ANDROID_BASE_URL/$ANDROID_NDK_FILENAME
ANDROID_NDK_PATH=$GODOT_BUILD_TOOLS_PATH/$ANDROID_NDK_DIR
ANDROID_NDK_SHA1=0bf02d4e8b85fd770fd7b9b2cdec57f9441f27a2

echo
echo "Download and install Android development tools ..."
echo

if [ ! -e $ANDROID_SDK_FILENAME ]; then
  echo "Downloading: Android SDK ..."
  curl -L -O $ANDROID_SDK_URL
else
  echo $ANDROID_SDK_SHA1 $ANDROID_SDK_FILENAME > $ANDROID_SDK_FILENAME.sha1
  if [ $(shasum -a 256 < $ANDROID_SDK_FILENAME | awk '{print $1;}') != $ANDROID_SDK_SHA1 ]; then
    echo "Downloading: Android SDK ..."
    curl -L -O $ANDROID_SDK_URL
  fi
fi

if [ ! -d $ANDROID_SDK_DIR ]; then
  echo "Extracting: Android SDK ..."
  unzip -qq $ANDROID_SDK_FILENAME -d $ANDROID_SDK_DIR
  echo
fi

if [ ! -e $ANDROID_NDK_FILENAME ]; then
  echo "Downloading: Android NDK ..."
  curl -L -O $ANDROID_NDK_URL
else
  echo $ANDROID_NDK_MD5 $ANDROID_NDK_FILENAME > $ANDROID_NDK_FILENAME.md5
  if [ $(shasum -a 1 < $ANDROID_NDK_FILENAME | awk '{print $1;}') != $ANDROID_NDK_SHA1 ]; then
    echo "Downloading: Android NDK ..."
    curl -L -O $ANDROID_NDK_URL
  fi
fi

if [ ! -d $ANDROID_NDK_DIR ]; then
  echo "Extracting: Android NDK ..."
  unzip -qq $ANDROID_NDK_FILENAME
  mv android-ndk-$ANDROID_NDK_RELEASE $ANDROID_NDK_DIR
  echo
fi

echo "Installing: Android Tools ..."
#$ANDROID_SDK_DIR/tools/bin/sdkmanager --all
yes | $ANDROID_SDK_DIR/tools/bin/sdkmanager --licenses > /dev/null
$ANDROID_SDK_DIR/tools/bin/sdkmanager 'tools' > /dev/null
$ANDROID_SDK_DIR/tools/bin/sdkmanager 'platform-tools' > /dev/null
$ANDROID_SDK_DIR/tools/bin/sdkmanager 'build-tools;26.0.2' > /dev/null
echo

EXPORT_VAL="export ANDROID_HOME=$ANDROID_SDK_PATH"
if ! grep -q "^$EXPORT_VAL" $BASH_RC; then
        echo $EXPORT_VAL >> $BASH_RC
fi
#eval $EXPORT_VAL

EXPORT_VAL="export ANDROID_NDK_ROOT=$ANDROID_NDK_PATH"
if ! grep -q "^$EXPORT_VAL" $BASH_RC; then
        echo $EXPORT_VAL >> $BASH_RC
fi
#eval $EXPORT_VAL

EXPORT_VAL="export PATH=$PATH:$ANDROID_SDK_PATH/tools"
if ! grep -q "^export PATH=.*$ANDROID_SDK_PATH/tools.*" $BASH_RC; then
        echo $EXPORT_VAL >> $BASH_RC
fi
#eval $EXPORT_VAL

EXPORT_VAL="export PATH=$PATH:$ANDROID_SDK_PATH/tools/bin"
if ! grep -q "^export PATH=.*$ANDROID_SDK_PATH/tools/bin.*" $BASH_RC; then
        echo $EXPORT_VAL >> $BASH_RC
fi
#eval $EXPORT_VAL

echo
echo "Done!"
echo
