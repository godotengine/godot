#include <include/syscall.hpp>
#include <include/printf.hpp>
#include <cstdint>
#include <cstdarg>

uint64_t __stack_chk_guard = 0x123456780C0A00FF;

extern "C"
__attribute__((noreturn))
void panic(const char* reason)
{
	tfp_printf("\n\n!!! PANIC !!!\n%s\n", reason);

	// the end
	syscall(SYSCALL_EXIT, -1);
	__builtin_unreachable();
}

extern "C"
void abort()
{
	panic("Abort called");
}

extern "C"
void abort_message(const char* fmt, ...)
{
	char buffer[2048];
	va_list arg;
	va_start (arg, fmt);
	int bytes = tfp_vsnprintf(buffer, sizeof(buffer), fmt, arg);
	(void) bytes;
	va_end (arg);
	panic(buffer);
}

extern "C"
void __assert_func(
	const char *file,
	int line,
	const char *func,
	const char *failedexpr)
{
	tfp_printf(
		"assertion \"%s\" failed: file \"%s\", line %d%s%s\n",
		failedexpr, file, line,
		func ? ", function: " : "", func ? func : "");
	abort();
}

extern "C"
void __stack_chk_fail()
{
	panic("Stack protector failed check");
}
