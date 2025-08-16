export DEBUG=1
source build.sh
echo "=================================="
echo " Start emulator with DEBUG=1 now. "
echo "=================================="
gdb-multiarch -x debug.gdb
