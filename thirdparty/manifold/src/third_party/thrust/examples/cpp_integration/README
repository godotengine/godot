This example shows how to link a Thrust program contained in 
a .cu file with a C++ program contained in a .cpp file.  Note
that device_vector only appears in the .cu file while host_vector
appears in both.  This relects the fact that algorithms on device
vectors are only available when the contents of the program are
located in a .cu file and compiled with the nvcc compiler.

On a Linux system where Thrust is installed in the default location
we can use the following procedure to compile the two parts of the
program and link them together.

  $ nvcc -O2 -c device.cu
  $ g++  -O2 -c host.cpp   -I/usr/local/cuda/include/
  $ nvcc -o tester device.o host.o

Alternatively, we can use g++ to perform final linking step.

  $ nvcc -O2 -c device.cu
  $ g++  -O2 -c host.cpp   -I/usr/local/cuda/include/
  $ g++ -o tester device.o host.o -L/usr/local/cuda/lib64 -lcudart

