Directions for compiling and running the benchmark with Ubuntu Linux:

Install Intel's Threading Building Blocks library (TBB):
$ sudo apt-get install libtbb-dev

Compile the benchmark:
$ nvcc -O3 -arch=sm_20 bench.cu -ltbb -o bench

Run the benchmark:
$ ./bench

Typical output (Tesla C2050):

Benchmarking with input size 33554432
Core Primitive Performance (elements per second)
      Algorithm,          STL,          TBB,       Thrust
         reduce,   3121746688,   3739585536,  26134038528
      transform,   1869492736,   2347719424,  13804681216
           scan,   1394143744,   1439394816,   5039195648
           sort,     11070660,     34622352,    673543168
Sorting Performance (keys per second)
  Type,          STL,          TBB,       Thrust
  char,     24050078,     62987040,   2798874368
 short,     15644141,     41275164,   1428603008
   int,     11062616,     33478628,    682295744
  long,     11249874,     33972564,    219719184
 float,      9850043,     29011806,    692407232
double,      9700181,     27153626,    224345568

The reported numbers are performance rates in "elements per second" (higher is better).

