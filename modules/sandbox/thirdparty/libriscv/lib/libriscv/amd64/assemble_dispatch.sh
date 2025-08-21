set -e
# An assembler variation of the RISC-V emulator dispatch function
nasm -gdwarf -O3 -f elf64 inaccurate_dispatch.nasm -o inaccurate_dispatch_rv64gb.o
if [ $? -ne 0 ]; then
	echo "Error: Assembly failed."
	exit 1
fi
