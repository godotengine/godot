#include <stddef.h>
extern long read(int fd, char* data, size_t maxlen);
extern long write(int fd, const char* data, size_t len);
extern __attribute__((noreturn)) void exit(long status);

static void strcpy(char* dst, const char* src) {
	while (*src) {
		*dst++ = *src++;
	}
	*dst = 0;
}
static size_t strlen(const char* dst) {
	const char* src = dst;
	while (*dst) dst++;
	return dst - src;
}

// Demo: Type "Hello World!" in the terminal.
long main(int argc, char** argv)
{
	char data[1024];
	return write(1, data,
		read(0, data, sizeof(data))
	);
}
