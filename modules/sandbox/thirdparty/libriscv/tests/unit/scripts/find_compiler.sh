if command -v "riscv64-linux-gnu-gcc-12" &> /dev/null
then
    echo "Found RISC-V compiler: GCC 12"
    export RCC="riscv64-linux-gnu-gcc-12"
    export RCXX="riscv64-linux-gnu-g++-12"

elif command -v "riscv64-linux-gnu-gcc-11" &> /dev/null
then
    echo "Found RISC-V compiler: GCC 11"
    export RCC="riscv64-linux-gnu-gcc-11"
    export RCXX="riscv64-linux-gnu-g++-11"

elif command -v "riscv64-linux-gnu-gcc-10" &> /dev/null
then
    echo "Found RISC-V compiler: GCC 10"
    export RCC="riscv64-linux-gnu-gcc-10"
    export RCXX="riscv64-linux-gnu-g++-10"
fi

#export RCC="riscv64-unknown-elf-gcc"
#export RCXX="riscv64-unknown-elf-g++"

#export RCC="zig cc -target riscv64-linux-musl -mcpu=baseline_rv64+rva22u64"
#export RCXX="zig c++ -target riscv64-linux-musl -mcpu=baseline_rv64+rva22u64"
