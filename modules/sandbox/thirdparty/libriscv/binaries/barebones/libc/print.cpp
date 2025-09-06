#include <cassert>
#include <cstdarg>
#include <stdio.h>
#include <include/libc.hpp>
#include <include/printf.hpp>
#include <include/syscall.hpp>

extern "C"
long sys_write(const void* data, size_t len)
{
	return syscall(SYSCALL_WRITE, 1, (long) data, len);
}

#undef printf
__attribute__((format (printf, 1, 2)))
int printf(const char* fmt, ...)
{
	char buffer[4096];
	va_list va;
	va_start(va, fmt);
	int len = tfp_vsnprintf(buffer, sizeof(buffer), fmt, va);
	va_end(va);

	return sys_write(buffer, len);
}

#undef snprintf
__attribute__((format (printf, 3, 4)))
int snprintf(char *s, size_t maxlen, const char *format, ...)
{
	va_list arg;
	va_start (arg, format);
	int bytes = tfp_vsnprintf(s, maxlen, format, arg);
	va_end (arg);
	return bytes;
}

#undef fprintf
#undef vfprintf
int vfprintf(FILE* fp, const char *format, va_list ap)
{
	(void) fp;
	char buffer[4096];
	int len = tfp_vsnprintf(buffer, sizeof(buffer), format, ap);
	sys_write(buffer, len);
	return len;
}

__attribute__((format (printf, 2, 3)))
int fprintf(FILE* stream, const char* fmt, ...)
{
	va_list arg;
    va_start (arg, fmt);
    int bytes = vfprintf(stream, fmt, arg);
    va_end (arg);
	return bytes;
}

__attribute__((format(printf, 2, 3)))
int __printf_chk (int flag, const char *format, ...)
{
	(void) flag;
	va_list ap;
	va_start (ap, format);
	int bytes = vfprintf (stdout, format, ap);
	va_end (ap);
	return bytes;
}
int __fprintf_chk(FILE* fp, int flag, const char* format, ...)
{
	(void) flag;
	va_list arg;
	va_start (arg, format);
	int bytes = vfprintf(fp, format, arg);
	va_end (arg);
	return bytes;
}
int __vfprintf_chk(FILE* fp, int flag, const char *format, va_list ap)
{
	(void) flag;
	int bytes = vfprintf (fp, format, ap);
	return bytes;
}
int __vsprintf_chk(char* s, int flag, size_t slen, const char* format, va_list args)
{
	(void) flag;
	int res = tfp_vsnprintf(s, slen, format, args);
	assert ((size_t) res < slen);
	return res;
}
int __vsnprintf_chk (char *s, size_t maxlen, int flags, size_t slen,
		                  const char *format, va_list args)
{
	assert (slen < maxlen);
	(void) flags;
	return tfp_vsnprintf(s, slen, format, args);
}
__attribute__((format(printf, 4, 5)))
int __sprintf_chk(char* s, int flags, size_t slen, const char *format, ...)
{
	va_list arg;
	va_start (arg, format);
	int bytes = __vsprintf_chk(s, flags, slen, format, arg);
	va_end (arg);
	return bytes;
}
int __snprintf_chk (char *s, size_t maxlen, int flags, size_t slen,
                		 const char *format, ...)
{
	va_list arg;
	int done;

	va_start (arg, format);
	done = __vsnprintf_chk (s, maxlen, flags, slen, format, arg);
	va_end (arg);

	return done;
}
