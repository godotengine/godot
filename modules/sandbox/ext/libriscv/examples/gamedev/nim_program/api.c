#include <stddef.h>

#define ECALL_WRITE  64
#define ECALL_EXIT   93

long fast_write(int fd, const char *buffer, size_t size)
{
	register int          a0 __asm__("a0") = fd;
	register const char*  a1 __asm__("a1") = buffer;
	register size_t       a2 __asm__("a2") = size;
	register long syscall_id __asm__("a7") = ECALL_WRITE;

	__asm__ volatile("ecall"
				 : "+r"(a0)
				 : "m"(*(const char(*)[size])a1), "r"(a2),
				   "r"(syscall_id));
	return a0;
}

void fast_exit(int status)
{
	__asm__ volatile("wfi\nj fast_exit\n");
	__builtin_unreachable();
}

void measure_overhead() {}


#define DEFINE_DYNCALL(number, name) \
	__asm__(".pushsection .text\n" \
		".global " #name "\n" \
		".type " #name ", @function\n" \
		"" #name ":\n" \
		"   .insn i 0b1011011, 0, x0, x0, " #number "\n" \
		"   ret\n"   \
		".popsection .text\n"); \
	extern void name() __attribute__((used, retain));

DEFINE_DYNCALL(1, dyncall1);
DEFINE_DYNCALL(2, dyncall2);
DEFINE_DYNCALL(3, dyncall3);
DEFINE_DYNCALL(4, dyncall4);
DEFINE_DYNCALL(5, dyncall5);
DEFINE_DYNCALL(6, dyncall6);
DEFINE_DYNCALL(7, dyncall7);
DEFINE_DYNCALL(8, dyncall8);
DEFINE_DYNCALL(9, dyncall9);
DEFINE_DYNCALL(10, dyncall10);
DEFINE_DYNCALL(11, dyncall11);
DEFINE_DYNCALL(12, dyncall12);
DEFINE_DYNCALL(13, dyncall13);
DEFINE_DYNCALL(14, dyncall14);
DEFINE_DYNCALL(15, dyncall15);
DEFINE_DYNCALL(16, dyncall16);
DEFINE_DYNCALL(17, dyncall17);
DEFINE_DYNCALL(18, dyncall18);
DEFINE_DYNCALL(19, dyncall19);
DEFINE_DYNCALL(20, dyncall20);
DEFINE_DYNCALL(21, dyncall21);
DEFINE_DYNCALL(22, dyncall22);
DEFINE_DYNCALL(23, dyncall23);
DEFINE_DYNCALL(24, dyncall24);
DEFINE_DYNCALL(25, dyncall25);
DEFINE_DYNCALL(26, dyncall26);
DEFINE_DYNCALL(27, dyncall27);
DEFINE_DYNCALL(28, dyncall28);
DEFINE_DYNCALL(29, dyncall29);
DEFINE_DYNCALL(30, dyncall30);
DEFINE_DYNCALL(31, dyncall31);
DEFINE_DYNCALL(32, dyncall32);

#define NATIVE_MEM_FUNCATTR		 /* */
#define NATIVE_SYSCALLS_BASE 490 /* libc starts at 490 */

#define SYSCALL_MALLOC  (NATIVE_SYSCALLS_BASE + 0)
#define SYSCALL_CALLOC  (NATIVE_SYSCALLS_BASE + 1)
#define SYSCALL_REALLOC (NATIVE_SYSCALLS_BASE + 2)
#define SYSCALL_FREE    (NATIVE_SYSCALLS_BASE + 3)
#define SYSCALL_MEMINFO (NATIVE_SYSCALLS_BASE + 4)

#define SYSCALL_MEMCPY  (NATIVE_SYSCALLS_BASE + 5)
#define SYSCALL_MEMSET  (NATIVE_SYSCALLS_BASE + 6)
#define SYSCALL_MEMMOVE (NATIVE_SYSCALLS_BASE + 7)
#define SYSCALL_MEMCMP  (NATIVE_SYSCALLS_BASE + 8)

#define SYSCALL_STRLEN  (NATIVE_SYSCALLS_BASE + 10)
#define SYSCALL_STRCMP  (NATIVE_SYSCALLS_BASE + 11)

#define SYSCALL_BACKTRACE (NATIVE_SYSCALLS_BASE + 19)


void *__wrap_malloc(size_t size)
{
	register void *ret __asm__("a0");
	register size_t a0 __asm__("a0") = size;
	register long syscall_id __asm__("a7") = SYSCALL_MALLOC;

	asm volatile("ecall"
				 : "=m"(*(char(*)[size])ret), "=r"(ret)
				 : "r"(a0), "r"(syscall_id));
	return ret;
}

void __wrap_free(void *ptr)
{
	register void *a0 __asm__("a0") = ptr;
	register long syscall_id __asm__("a7") = SYSCALL_FREE;

	asm volatile("ecall"
				 :
				 : "r"(a0), "r"(syscall_id));
}

#define STR1(x) #x
#define STR(x) STR1(x)

__asm__(".pushsection .text\n"
	".global __wrap_calloc\n"
	".type __wrap_calloc, @function\n"
	"__wrap_calloc:\n"
	"	li a7, " STR(SYSCALL_CALLOC) "\n"
	"	ecall\n"
	"	ret\n"
	".global __wrap_realloc\n"
	".type __wrap_realloc, @function\n"
	"__wrap_realloc:\n"
	"	li a7, " STR(SYSCALL_REALLOC) "\n"
	"	ecall\n"
	"	ret\n"
	".popsection .text\n");


void* __wrap_memset(void* vdest, const int ch, size_t size)
{
	register char*   a0 __asm__("a0") = (char*)vdest;
	register int     a1 __asm__("a1") = ch;
	register size_t  a2 __asm__("a2") = size;
	register long syscall_id __asm__("a7") = SYSCALL_MEMSET;

	asm volatile ("ecall"
	:	"=m"(*(char(*)[size]) a0)
	:	"r"(a0), "r"(a1), "r"(a2), "r"(syscall_id));
	return vdest;
}

void* __wrap_memcpy(void* vdest, const void* vsrc, size_t size)
{
	register char*       a0 __asm__("a0") = (char*)vdest;
	register const char* a1 __asm__("a1") = (const char*)vsrc;
	register size_t      a2 __asm__("a2") = size;
	register long syscall_id __asm__("a7") = SYSCALL_MEMCPY;

	asm volatile ("ecall"
	:	"=m"(*(char(*)[size]) a0), "+r"(a0)
	:	"r"(a1), "m"(*(const char(*)[size]) a1),
		"r"(a2), "r"(syscall_id));
	return vdest;
}

void* __wrap_memmove(void* vdest, const void* vsrc, size_t size)
{
	// An assumption is being made here that since vsrc might be
	// inside vdest, we cannot assume that vsrc is const anymore.
	register char*  a0 __asm__("a0") = (char*)vdest;
	register char*  a1 __asm__("a1") = (char*)vsrc;
	register size_t a2 __asm__("a2") = size;
	register long syscall_id __asm__("a7") = SYSCALL_MEMMOVE;

	asm volatile ("ecall"
		: "=m"(*(char(*)[size]) a0), "=m"(*(char(*)[size]) a1)
		: "r"(a0), "r"(a1), "r"(a2), "r"(syscall_id));
	return vdest;
}

int __wrap_memcmp(const void* m1, const void* m2, size_t size)
{
	register const char* a0 __asm__("a0") = (const char*)m1;
	register const char* a1 __asm__("a1") = (const char*)m2;
	register size_t      a2 __asm__("a2") = size;
	register long syscall_id __asm__("a7") = SYSCALL_MEMCMP;
	register int         a0_out __asm__("a0");

	asm volatile ("ecall" : "=r"(a0_out) :
		"r"(a0), "m"(*(const char(*)[size]) a0),
		"r"(a1), "m"(*(const char(*)[size]) a1),
		"r"(a2), "r"(syscall_id));
	return a0_out;
}

size_t __wrap_strlen(const char* str)
{
	register const char* a0 __asm__("a0") = str;
	register size_t      a0_out __asm__("a0");
	register long syscall_id __asm__("a7") = SYSCALL_STRLEN;

	asm volatile ("ecall" : "=r"(a0_out) :
		"r"(a0), "m"(*(const char(*)[4096]) a0), "r"(syscall_id));
	return a0_out;
}

int __wrap_strcmp(const char* str1, const char* str2)
{
	register const char* a0 __asm__("a0") = str1;
	register const char* a1 __asm__("a1") = str2;
	register size_t      a2 __asm__("a2") = 4096;
	register size_t      a0_out __asm__("a0");
	register long syscall_id __asm__("a7") = SYSCALL_STRCMP;

	asm volatile ("ecall" : "=r"(a0_out) :
		"r"(a0), "m"(*(const char(*)[4096]) a0),
		"r"(a1), "m"(*(const char(*)[4096]) a1),
		"r"(a2), "r"(syscall_id));
	return a0_out;
}

int __wrap_strncmp(const char* str1, const char* str2, size_t maxlen)
{
	register const char* a0 __asm__("a0") = str1;
	register const char* a1 __asm__("a1") = str2;
	register size_t      a2 __asm__("a2") = maxlen;
	register size_t      a0_out __asm__("a0");
	register long syscall_id __asm__("a7") = SYSCALL_STRCMP;

	asm volatile ("ecall" : "=r"(a0_out) :
		"r"(a0), "m"(*(const char(*)[maxlen]) a0),
		"r"(a1), "m"(*(const char(*)[maxlen]) a1),
		"r"(a2), "r"(syscall_id));
	return a0_out;
}
