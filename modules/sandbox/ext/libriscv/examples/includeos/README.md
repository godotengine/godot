# RISC-V emulator hosted on IncludeOS

Setting up the build folder:
```
./setup_build.sh
```

Building and running the service:
```
cd build
source activate.sh
make -j && boot --create-bridge riscvemu
```

Sending a binary to run in the VM guest:
```
./send_file.sh ../../binaries/barebones/build/hello_world 10.0.0.42 1234
./send_file.sh ../../binaries/micro/build/hello_world 10.0.0.42 1234
./send_file.sh ../../binaries/newlib/build/hello_world 10.0.0.42 1234
./send_file.sh ../../binaries/full/build/hello_world 10.0.0.42 1234
```

Maximum inception achieved! You should see some output like this:

```
CPU 2 TID 4 executing 5226976 bytes binary
* Loading program of size 440794 from 0x1800020 to virtual 0x10000
* Program segment readable: 1 writable: 0  executable: 1
* Loading program of size 7784 from 0x186c1b4 to virtual 0x7d194
* Program segment readable: 1 writable: 1  executable: 0
* Entry is at 0x107b0
* SP = 0x3FFFFF00  Argument list: 168 bytes
* Program end: 0x7EFFC
>>> Warning: Unhandled syscall 48
>>> Warning: Unhandled syscall 134
>>> Warning: Unhandled syscall 134
>>> Warning: Unhandled syscall 163
Received exception from machine: Execution space protection fault
* Executed 743900 instructions in 17033 micros
* Machine output:
Hello, Global Constructor!
Hello RISC-V World v1.0!
...
```
