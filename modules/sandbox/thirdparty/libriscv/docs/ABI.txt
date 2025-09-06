## Registers

x0     ZERO
x1 RA  Return address
x2 SP  Stack pointer
x3 GP  Global pointer
x4 TP  Thread pointer
x5 LR  Link register
x6-7   Temporaries
x8 S0  Saved frame pointer
x9 S1  Saved register
x10-11 Function args & return values
x12-17 Function arguments
x18-27 Saved registers
x28-31 Temporaries

a0-a7  Function arguments (x10-x17)
a0-a1  Function return values

a0-a6  System call arguments
a7     System call number
a0     System call return value

f0-7   FP temporaries
f8-9   FP saved registers
f10-11 FP args & return values
f12-17 FP arguments
f18-27 FP saved registers
f28-31 FP temporaries

PC     Program counter

## Alignments and trivia

All instructions 4-byte aligned, except for C-extension where instructions are 2-byte aligned. It is even allowed to have a 2-byte aligned 4-byte instruction cross a page boundary, unfortunately.

The stack pointer must be 16-byte aligned.

GP must be initialized to some symbol (`__global_pointer`) in the middle of the data section. This happens right after _start in most run-times. Golang does not use GP this way.
