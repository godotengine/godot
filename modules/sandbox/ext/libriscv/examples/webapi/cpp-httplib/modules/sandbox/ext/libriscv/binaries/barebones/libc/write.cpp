#include <cassert>
#include <cstring>
#include <cstdio>
#include <include/syscall.hpp>

extern "C"
long write(int fd, const void* data, size_t len)
{
	return syscall(SYSCALL_WRITE, fd, (long) data, len);
}

extern "C"
long _write_r(_reent*, int fd, const void* data, size_t len)
{
	return syscall(SYSCALL_WRITE, fd, (long) data, len);
}

extern "C"
int puts(const char* string)
{
	const long len = __builtin_strlen(string);
	return write(0, string, len);
}

// buffered serial output
static char buffer[256];
static unsigned cnt = 0;

extern "C"
int fflush(FILE*)
{
	long ret = write(0, buffer, cnt);
	cnt = 0;
	return ret;
}

extern "C"
void __print_putchr(void*, char c)
{
	buffer[cnt++] = c;
	if (c == '\n' || cnt == sizeof(buffer)) {
		fflush(0);
	}
}
