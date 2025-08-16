#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef NATIVE_MEM_SYSCALLS
#define NATIVE_MEM_FUNCATTR /* */
#include <include/syscall.hpp>
#define SYS_IARRAY *(const char(*)[size])
#define SYS_OARRAY *(char(*)[size])
#define SYS_IUARRAY *(const char*)
#define SYS_OUARRAY *(char*)
#else
#define NATIVE_MEM_FUNCATTR /* */
#endif
#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

#ifndef USE_NEWLIB
extern "C" struct _reent* _impure_ptr;
void* __dso_handle;

extern "C" int* __errno() {
	static int errno_value = 0;
	return &errno_value;
}
#endif

#ifndef NATIVE_MEM_SYSCALLS
extern "C"
void* memset(void* vdest, int ch, size_t size)
{
	char* dest = (char*) vdest;
	for (size_t i = 0; i < size; i++)
		dest[i] = ch;
	return dest;
}
void* memcpy(void* vdest, const void* vsrc, size_t size)
{
	const char* src = (const char*) vsrc;
	char* dest = (char*) vdest;
	for (size_t i = 0; i < size; i++)
		dest[i] = src[i];
	return vdest;
}
void* memmove(void* vdest, const void* vsrc, size_t size)
{
	const char* src = (const char*) vsrc;
	char* dest = (char*) vdest;
	if (dest <= src)
	{
		for (size_t i = 0; i < size; i++)
			dest[i] = src[i];
	}
	else
	{
		for (int i = size-1; i >= 0; i--)
			dest[i] = src[i];
	}
	return dest;
}

#else

#if defined(USE_NEWLIB) && defined(WRAP_NATIVE_SYSCALLS)
#define memset  __wrap_memset
#define memcpy  __wrap_memcpy
#define memmove __wrap_memmove
#define memcmp  __wrap_memcmp

#define strlen  __wrap_strlen
#define strcmp  __wrap_strcmp
#define strncmp __wrap_strncmp
#endif

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

#endif


#ifndef USE_NEWLIB

extern "C"
wchar_t* wmemcpy(wchar_t* wto, const wchar_t* wfrom, size_t size)
{
	return (wchar_t *) memcpy (wto, wfrom, size * sizeof (wchar_t));
}

extern "C"
void* memchr(const void *s, int c, size_t n)
{
    if (n != 0) {
        const auto* p = (const unsigned char*) s;

        do {
            if (*p++ == c)
                return ((void *)(p - 1));
        } while (--n != 0);
    }
    return nullptr;
}

extern "C"
char* strcpy(char* dst, const char* src)
{
	while ((*dst++ = *src++));
	*dst = 0;
	return dst;
}
#ifndef NATIVE_MEM_SYSCALLS
extern "C"
size_t strlen(const char* str)
{
	const char* iter;
	for (iter = str; *iter; ++iter);
	return iter - str;
}
extern "C"
int strcmp(const char* str1, const char* str2)
{
	while (*str1 != 0 && *str2 != 0  && *str1 == *str2) {
      str1++;
      str2++;
   }
   return *str1 - *str2;
}
extern "C"
int strncmp(const char* s1, const char* s2, size_t n)
{
    while ( n && *s1 && ( *s1 == *s2 ) )
    {
        ++s1;
        ++s2;
        --n;
    }
    if ( n == 0 )
    {
        return 0;
    }
    else
    {
        return ( *(unsigned char *)s1 - *(unsigned char *)s2 );
    }
}
#endif

extern "C"
char* strcat(char* dest, const char* src)
{
	strcpy(dest + strlen(dest), src);
	return dest;
}

extern "C"
void* __memcpy_chk(void* dest, const void* src, size_t len, size_t destlen)
{
  assert (len <= destlen);
  return memcpy(dest, src, len);
}
extern "C"
void* __memset_chk(void* dest, int c, size_t len, size_t destlen)
{
  assert (len <= destlen);
  return memset(dest, c, len);
}
extern "C"
char* __strcat_chk(char* dest, const char* src, size_t destlen)
{
  size_t len = strlen(dest) + strlen(src) + 1;
  assert (len <= destlen);
  return strcat(dest, src);
}

extern "C"
int abs(int value)
{
	return (value >= 0) ? value : -value;
}

#endif

extern "C" __attribute__((noreturn))
void _exit(int code)
{
	register long a0 asm("a0") = code;

	asm volatile("r%=: wfi \nj r%=\n" :: "r"(a0));
	__builtin_unreachable();
}

extern "C" __attribute__((noreturn))
void __wrap_exit(int code)
{
	_exit(code);
}
