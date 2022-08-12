#BUILD THE GODOT ENGINE:
set -e

BASE_DIR="$( cd "$(dirname "$0")" ; pwd -P )"

cd $BASE_DIR
scons -j20 platform=linux target=debug