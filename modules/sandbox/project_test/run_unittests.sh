set -e
GODOT_VERSION=4.4.1

# Check if unit tests are run from Github Actions
if [ -n "$CI" ]; then
	GODOT=./Godot_v4.4.1-stable_linux.x86_64
	# Use the --import flag to properly initialize the project
	$GODOT --path "$PWD" --headless --import
else
	# Use a local Godot binary
	if [ -z "$GODOT" ]; then
		GODOT=~/Godot_v4.4.1-stable_linux.x86_64
	fi
fi

export CXX="riscv64-linux-gnu-g++-14"
#export CXX="riscv64-unknown-elf-g++"

# Build the unit test ELF file
mkdir -p .build
pushd .build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake -DSTRIPPED=OFF -DFLTO=ON
make -j4
popd

# Create a symbolic link to the unit test ELF file
ln -fs ../.build/unittests tests/tests.elf

# Import again for CI
if [ -n "$CI" ]; then
	$GODOT --path "$PWD" --headless --import
fi

# Run the unit tests using the GUT addon
$GODOT --path "$PWD" --headless -s addons/gut/gut_cmdln.gd $@
