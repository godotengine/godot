#include <include/syscall.hpp>
#include <tinyprintf.h>
#include <stdlib.h>
#include <cstdint>
static struct _reent reent;
struct _reent* _impure_ptr = &reent;
__FILE __sf[3];

extern "C" {
	__attribute__((noreturn))
	void _exit(int exitval);
	void __print_putchr(void* file, char c);
}

static void
init_stdlib()
{
	_REENT_INIT_PTR_ZEROED(_impure_ptr);

	// 2. Enable printf facilities
	init_printf(NULL, __print_putchr);

	// 3. Call global C/C++ constructors
	extern void(*__init_array_start [])();
	extern void(*__init_array_end [])();
	const int count = __init_array_end - __init_array_start;
	for (int i = 0; i < count; i++) {
		__init_array_start[i]();
	}
}

extern "C" __attribute__((visibility("hidden"), used))
void libc_start(int argc, char** argv, char** envp)
{
	init_stdlib();

	// Call main() :)
	extern int main(int, char**, char**);
	_exit(main(argc, argv, envp));
}

// 1. wrangle with argc and argc
// 2. initialize the global pointer to __global_pointer
// NOTE: have to disable relaxing first
asm
("  .global _start				\t\n\
_start:                         \t\n\
    lw   a0, 0(sp) 				\t\n"
#if __riscv_xlen == 32
	"addi a1, sp, 4		 		\t\n"
#else
	"addi a1, sp, 8		 		\t\n"
#endif
	"andi sp, sp, -16 /* not needed */\t\n\
.option push 					\t\n\
.option norelax 				\t\n\
	1:auipc gp, %pcrel_hi(__global_pointer$) \t\n\
	addi  gp, gp, %pcrel_lo(1b) \t\n\
.option pop						\t\n\
	call libc_start				\t\n\
");
