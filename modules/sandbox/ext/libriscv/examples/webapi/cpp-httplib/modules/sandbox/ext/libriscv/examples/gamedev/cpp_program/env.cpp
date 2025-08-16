#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#define NATIVE_MEM_FUNCATTR /* */
#define NATIVE_SYSCALLS_BASE  490  /* libc starts at 490 */

#define SYSCALL_MALLOC    (NATIVE_SYSCALLS_BASE+0)
#define SYSCALL_CALLOC    (NATIVE_SYSCALLS_BASE+1)
#define SYSCALL_REALLOC   (NATIVE_SYSCALLS_BASE+2)
#define SYSCALL_FREE      (NATIVE_SYSCALLS_BASE+3)
#define SYSCALL_MEMINFO   (NATIVE_SYSCALLS_BASE+4)

#define SYSCALL_MEMCPY    (NATIVE_SYSCALLS_BASE+5)
#define SYSCALL_MEMSET    (NATIVE_SYSCALLS_BASE+6)
#define SYSCALL_MEMMOVE   (NATIVE_SYSCALLS_BASE+7)
#define SYSCALL_MEMCMP    (NATIVE_SYSCALLS_BASE+8)

#define SYSCALL_STRLEN    (NATIVE_SYSCALLS_BASE+10)
#define SYSCALL_STRCMP    (NATIVE_SYSCALLS_BASE+11)

#define SYSCALL_BACKTRACE (NATIVE_SYSCALLS_BASE+19)


#define SYS_IARRAY *(const char(*)[size])
#define SYS_OARRAY *(char(*)[size])
#define SYS_IUARRAY *(const char*)
#define SYS_OUARRAY *(char*)
#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

#define memset  __wrap_memset
#define memcpy  __wrap_memcpy
#define memmove __wrap_memmove
#define memcmp  __wrap_memcmp

#define strlen  __wrap_strlen
#define strcmp  __wrap_strcmp
#define strncmp __wrap_strncmp

//-- Memory functions --//

extern "C" NATIVE_MEM_FUNCATTR
void* memset(void* vdest, const int ch, size_t size)
{
	register char*   a0 asm("a0") = (char*)vdest;
	register int     a1 asm("a1") = ch;
	register size_t  a2 asm("a2") = size;
	register long syscall_id asm("a7") = SYSCALL_MEMSET;

	asm volatile ("ecall"
	:	"=m"(*(char(*)[size]) a0)
	:	"r"(a0), "r"(a1), "r"(a2), "r"(syscall_id));
	return vdest;
}

extern "C"
void* memcpy(void* vdest, const void* vsrc, size_t size)
{
	register char*       a0 asm("a0") = (char*)vdest;
	register const char* a1 asm("a1") = (const char*)vsrc;
	register size_t      a2 asm("a2") = size;
	register long syscall_id asm("a7") = SYSCALL_MEMCPY;

	asm volatile ("ecall"
	:	"=m"(*(char(*)[size]) a0)
	:	"r"(a0),
		"r"(a1), "m"(*(const char(*)[size]) a1),
		"r"(a2), "r"(syscall_id));
	return vdest;
}

extern "C" NATIVE_MEM_FUNCATTR
void* memmove(void* vdest, const void* vsrc, size_t size)
{
	// An assumption is being made here that since vsrc might be
	// inside vdest, we cannot assume that vsrc is const anymore.
	register char*  a0 asm("a0") = (char*)vdest;
	register char*  a1 asm("a1") = (char*)vsrc;
	register size_t a2 asm("a2") = size;
	register long syscall_id asm("a7") = SYSCALL_MEMMOVE;

	asm volatile ("ecall"
		: "=m"(*(char(*)[size]) a0), "=m"(*(char(*)[size]) a1)
		: "r"(a0), "r"(a1), "r"(a2), "r"(syscall_id));
	return vdest;
}

extern "C" NATIVE_MEM_FUNCATTR
int memcmp(const void* m1, const void* m2, size_t size)
{
	register const char* a0 asm("a0") = (const char*)m1;
	register const char* a1 asm("a1") = (const char*)m2;
	register size_t      a2 asm("a2") = size;
	register long syscall_id asm("a7") = SYSCALL_MEMCMP;
	register int         a0_out asm("a0");

	asm volatile ("ecall" : "=r"(a0_out) :
		"r"(a0), "m"(*(const char(*)[size]) a0),
		"r"(a1), "m"(*(const char(*)[size]) a1),
		"r"(a2), "r"(syscall_id));
	return a0_out;
}

//-- String functions --//

extern "C" NATIVE_MEM_FUNCATTR
size_t strlen(const char* str)
{
	register const char* a0 asm("a0") = str;
	register size_t      a0_out asm("a0");
	register long syscall_id asm("a7") = SYSCALL_STRLEN;

	asm volatile ("ecall" : "=r"(a0_out) :
		"r"(a0), "m"(*(const char(*)[4096]) a0), "r"(syscall_id));
	return a0_out;
}

extern "C" NATIVE_MEM_FUNCATTR
int strcmp(const char* str1, const char* str2)
{
	register const char* a0 asm("a0") = str1;
	register const char* a1 asm("a1") = str2;
	register size_t      a2 asm("a2") = 4096;
	register size_t      a0_out asm("a0");
	register long syscall_id asm("a7") = SYSCALL_STRCMP;

	asm volatile ("ecall" : "=r"(a0_out) :
		"r"(a0), "m"(*(const char(*)[4096]) a0),
		"r"(a1), "m"(*(const char(*)[4096]) a1),
		"r"(a2), "r"(syscall_id));
	return a0_out;
}

extern "C" NATIVE_MEM_FUNCATTR
int strncmp(const char* str1, const char* str2, size_t maxlen)
{
	register const char* a0 asm("a0") = str1;
	register const char* a1 asm("a1") = str2;
	register size_t      a2 asm("a2") = maxlen;
	register size_t      a0_out asm("a0");
	register long syscall_id asm("a7") = SYSCALL_STRCMP;

	asm volatile ("ecall" : "=r"(a0_out) :
		"r"(a0), "m"(*(const char(*)[maxlen]) a0),
		"r"(a1), "m"(*(const char(*)[maxlen]) a1),
		"r"(a2), "r"(syscall_id));
	return a0_out;
}

//-- Heap memory management functions --//

#define GENERATE_SYSCALL_WRAPPER(name, number) \
	asm(".global " #name "\n" #name ":\n  li a7, " STRINGIFY(number) "\n  ecall\n  ret\n");

asm(".pushsection .text, \"ax\", @progbits\n");
GENERATE_SYSCALL_WRAPPER(__wrap_malloc,  SYSCALL_MALLOC);
GENERATE_SYSCALL_WRAPPER(__wrap_calloc,  SYSCALL_CALLOC);
GENERATE_SYSCALL_WRAPPER(__wrap_realloc, SYSCALL_REALLOC);
GENERATE_SYSCALL_WRAPPER(__wrap_free,    SYSCALL_FREE);
asm(".popsection\n");

//-- Exit functions --//

extern "C" __attribute__((noreturn, used, retain))
void fast_exit(int code)
{
	register long a0 asm("a0") = code;

	asm volatile("r%=: wfi \nj r%=\n" :: "r"(a0));
	__builtin_unreachable();
}

//-- Overhead measurement --//

extern "C" __attribute__((used, retain))
void measure_overhead() {}
