.text
.align 4

.globl _start
_start:
	lui   a5, 0xf424   # Load upper immediate with 256,000,000
	li    a0, 0
	li    a3, 1
loop:
	mv    a4, a0
	addi  a5, a5, -1
	add   a0, a4, a3
	mv    a3, a4
	bnez  a5, loop
	mv    a0, a4
end:
	li    a7, 93         # syscall for exit
	ecall                # make the syscall to exit
