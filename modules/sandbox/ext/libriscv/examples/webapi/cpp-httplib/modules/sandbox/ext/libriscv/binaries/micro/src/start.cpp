#include "syscall.hpp"
extern "C"
{
	__attribute__((noreturn))
	void _exit(int exitval) {
		syscall(SYSCALL_EXIT, exitval);
		__builtin_unreachable();
	}
	long write(int fd, const void* data, unsigned long len) {
		return syscall(SYSCALL_WRITE, fd, (long) data, len);
	}
}

extern "C" __attribute__((visibility("hidden"), used))
void libc_start(int argc, char** argv)
{
	// zero-initialize .bss section
	extern char __bss_start;
	extern char __BSS_END__;
	for (char* bss = &__bss_start; bss < &__BSS_END__; bss++) {
		*bss = 0;
	}
	asm volatile("" ::: "memory");

	// call global constructors
	extern void(*__init_array_start [])();
	extern void(*__init_array_end [])();
	int count = __init_array_end - __init_array_start;
	for (int i = 0; i < count; i++) {
		__init_array_start[i]();
	}

	// call main() :)
	extern int main(int, char**);
	_exit(main(argc, argv));
}

// 1. wrangle with argc and argc
// 2. initialize GP to __global_pointer
// NOTE: have to disable relaxing first
asm
("   .global _start             \t\n\
_start:                         \t\n\
     lw   a0, 0(sp) 			\t\n\
	 addi a1, sp, 4		 		\t\n\
	 andi sp, sp, -16 /* not needed */\t\n\
     .option push 				\t\n\
	 .option norelax 			\t\n\
	 1:auipc gp, %pcrel_hi(__global_pointer$) \t\n\
	 addi  gp, gp, %pcrel_lo(1b) \t\n\
	.option pop					\t\n\
	j libc_start				\t\n\
");
