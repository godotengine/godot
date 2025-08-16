#include <stddef.h>

#define ECALL_WRITE  64
#define ECALL_EXIT   93

int my_write(int fd, const char *buffer, size_t size)
{
	register int         a0 __asm__("a0")  = fd;
	register const char* a1 __asm__("a1")  = buffer;
	register size_t a2 __asm__("a2")	   = size;
	register long syscall_id __asm__("a7") = ECALL_WRITE;

	__asm__ volatile("ecall"
				 : "+r"(a0)
				 : "m"(*(const char(*)[size])a1), "r"(a2),
				   "r"(syscall_id));
	return a0;
}

int my_exit(int status)
{
	register int         a0 __asm__("a0")  = status;
	register long syscall_id __asm__("a7") = ECALL_EXIT;

	__asm__ volatile("ecall" : : "r"(a0), "r"(syscall_id));
	__builtin_unreachable();
}
