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

export CC="zig;cc;-target riscv64-linux-musl"
export CXX="zig;c++;-target riscv64-linux-musl"

# Build the unit test ELF file
mkdir -p .zig
pushd .zig
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DFLTO=ON
make -j4
popd

# Create a symbolic link to the unit test ELF file
ln -fs ../.zig/unittests tests/tests.elf

# Import again for CI
if [ -n "$CI" ]; then
	$GODOT --path "$PWD" --headless --import
fi

# Run the unit tests using the GUT addon
$GODOT --path "$PWD" --headless -s addons/gut/gut_cmdln.gd $@
