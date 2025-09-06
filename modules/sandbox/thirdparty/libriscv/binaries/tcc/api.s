.global read
.global write
.global exit
.global _start

.section .text
read:
	li a7, 63
	ecall
	ret
write:
	li a7, 64
	ecall
	ret

exit:
	li a7, 93
	ecall

# Startup -> main -> exit
_start:
	 lw   a0, 0(sp)
	 addi a1, sp, 8
	 call main
# .option push
# .option norelax
#	 1:auipc gp, %pcrel_hi(__global_pointer$)
#	 addi  gp, gp, %pcrel_lo(1b)
# .option pop
	 j exit
