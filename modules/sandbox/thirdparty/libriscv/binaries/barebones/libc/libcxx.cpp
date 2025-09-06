#include <cstddef>
extern "C"
__attribute__((noreturn)) void abort_message(const char* fmt, ...);

#ifndef USE_NEWLIB
#ifndef __EXCEPTIONS
// exception stubs for various C++ containers
namespace std {
	void __throw_bad_alloc() {
		abort_message("exception: bad_alloc thrown\n");
	}
	void __throw_length_error(char const*) {
		abort_message("C++ length error exception");
	}
	void __throw_bad_array_new_length() {
		abort_message("C++ bad array new length exception");
	}
	void __throw_logic_error(char const*) {
		abort_message("C++ length error exception");
	}
	void __throw_out_of_range_fmt(char const*, ...) {
		abort_message("C++ out-of-range exception");
	}
	void __throw_bad_function_call() {
		abort_message("Bad std::function call!");
	}
}
#endif

extern "C"
int __cxa_atexit(void (*func) (void*), void* /*arg*/, void* /*dso_handle*/)
{
	(void) func;
	return 0;
}
#endif
