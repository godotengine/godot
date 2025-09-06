BOLT=~/github/llvm-project/build/bin/llvm-bolt
PROG=rvlinux

$BOLT -instrument .build/$PROG -o $PROG.i
#./$PROG.i ../binaries/measure_mips/fib
#./$PROG.i ../binaries/STREAM/stream-rv64gb
./$PROG.i ~/github/coremark/coremark-rv32g_b
$BOLT $PROG -o $PROG.bolt -data=/tmp/prof.fdata -reorder-blocks=ext-tsp -reorder-functions=hfsort -split-functions -split-all-cold -split-eh -dyno-stats
