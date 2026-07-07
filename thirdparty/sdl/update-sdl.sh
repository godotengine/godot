#!/bin/bash -e

VERSION=3.2.28

target=$(dirname "$(realpath $0)")
pushd $target

rm -rf atomic core events haptic hidapi include io joystick libm loadso \
  sensor stdlib thread timer *.c *.h CREDITS.md LICENSE.txt
rm -rf *.tar.gz tmp

mkdir tmp && pushd tmp

echo "Updating SDL3 to version:" $VERSION
curl -L -O https://github.com/libsdl-org/SDL/archive/release-$VERSION.tar.gz

tar --strip-components=1 -xvf release-$VERSION.tar.gz
rm release-$VERSION.tar.gz

# We aim to copy only the minimum amount of files needed, so we don't need to
# vendor and compile more source code than necessary.

cp -v CREDITS.md LICENSE.txt $target

# Includes.
# For build config, we use a single private one in the driver.
# We might reconsider as we make more platforms use SDL.
cp -rv include $target
rm -f $target/include/build_config/{*.cmake,SDL_build_config_*.h} $target
rm -f $target/include/SDL3/SDL_{egl,gpu,oldnames,opengl*,test*,vulkan}.h $target

pushd src

# Shared dependencies

cp -rv *.{c,h} atomic libm stdlib $target
rm -f $target/stdlib/*.masm

# Only some files needed

mkdir $target/events
cp -v events/SDL_{event*.{c,h},mouse_c.h} $target/events

mkdir $target/io
cp -v io/SDL_iostream*.{c,h} $target/io

# Platform specific

mkdir $target/core
cp -rv core/{linux,unix,windows} $target/core
rm -f $target/core/windows/version.rc
rm -f $target/core/linux/SDL_{fcitx,ibus,ime,system_theme}.*

mkdir $target/haptic
cp -rv haptic/{*.{c,h},darwin,dummy,linux,windows} $target/haptic

mkdir $target/joystick
cp -rv joystick/{*.{c,h},apple,darwin,hidapi,linux,windows} $target/joystick

mkdir $target/loadso
cp -rv loadso/{dlopen,dummy} $target/loadso

mkdir $target/sensor
cp -rv sensor/{*.{c,h},dummy,windows} $target/sensor

mkdir $target/thread
cp -rv thread/{*.{c,h},pthread,windows} $target/thread
# Despite being 'generic', syssem.c is included in the Unix driver for macOS,
# and syscond/sysrwlock are used by the Windows driver.
# systhread_c.h is included by all these, but we should NOT compile the matching .c file.
mkdir $target/thread/generic
cp -v thread/generic/SDL_{syssem.c,{syscond,sysrwlock}*.{c,h},systhread_c.h} $target/thread/generic

mkdir $target/timer
cp -rv timer/{*.{c,h},unix,windows} $target/timer

# HIDAPI

mkdir -p $target/hidapi
cp -v hidapi/{*.{c,h},AUTHORS.txt,LICENSE.txt,LICENSE-bsd.txt,VERSION} $target/hidapi
for dir in hidapi linux mac windows; do
  mkdir $target/hidapi/$dir
  cp -v hidapi/$dir/*.{c,h} $target/hidapi/$dir
done

popd
popd
rm -rf tmp
popd

echo "SDL3 source code copied successfully. Review 'git status' for potential new files to compile or remove."
echo "Make sure to re-apply patches from the 'patches' folder if any are provided."
