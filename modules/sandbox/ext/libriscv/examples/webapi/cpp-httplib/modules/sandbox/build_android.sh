set -e
PROJECT="/sdcard/Documents/new-game-project"
VERSION="24.0.8215888"
# ANDROID_HOME and ANDROID_NDK_ROOT are already set in .bashrc
if [ ! -d "$ANDROID_NDK_ROOT" ]; then
	echo "Error: ANDROID_NDK_ROOT directory does not exist: $ANDROID_NDK_ROOT"
	exit 1
fi
if [ ! -d "$ANDROID_HOME" ]; then
	echo "Error: ANDROID_HOME directory does not exist: $ANDROID_HOME"
	exit 1
fi

## For --debug argument:
if [ "$1" == "--debug" ]; then
	echo "Building in debug mode..."
	scons platform=android target=template_debug debug_symbols=yes ndk_version=24.0.8215888
	#adb push bin/addons/godot_sandbox/bin/libgodot_riscv.android.template_release.arm64.so $PROJECT/addons/godot_sandbox/bin/libgodot_riscv.android.template_release.arm64.so
	cp bin/addons/godot_sandbox/bin/libgodot_riscv.android.template_debug.arm64.so bin/addons/godot_sandbox/bin/libgodot_riscv.android.template_release.arm64.so
	adb push bin/addons/godot_sandbox/bin/libgodot_riscv.android.template_debug.arm64.so $PROJECT/addons/godot_sandbox/bin/libgodot_riscv.android.template_release.arm64.so
## For --cmake build:
elif [ "$1" == "--cmake" ]; then
	echo "Building with CMake..."
	mkdir -p .build_android
	pushd .build_android
	cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
	      -DANDROID_ABI=arm64-v8a \
	      -DANDROID_PLATFORM=android-24 \
	      -DANDROID_STL=c++_shared \
	      -DANDROID_NDK=$ANDROID_NDK_ROOT \
	      -DANDROID_NATIVE_API_LEVEL=24 \
	      -DCMAKE_BUILD_TYPE=Release \
		  -DRISCV_LIBTCC=ON -DANDROID_TOOLCHAIN=ON \
		  ..
	make -j$(nproc)
	popd
	# This produces .build_android/libgodot-riscv.so
	adb push .build_android/libgodot-riscv.so $PROJECT/addons/godot_sandbox/bin/libgodot_riscv.android.template_release.arm64.so
## For --backtrace:
elif [ "$1" == "--backtrace" ]; then
	adb logcat | /usr/lib/android-ndk/ndk-stack -sym bin/addons/godot_sandbox/bin
else
## Default case: build in release mode
	echo "Building in release mode..."
	scons platform=android target=template_release ndk_version=24.0.8215888
	adb push bin/addons/godot_sandbox/bin/libgodot_riscv.android.template_release.arm64.so $PROJECT/addons/godot_sandbox/bin/libgodot_riscv.android.template_release.arm64.so
fi
