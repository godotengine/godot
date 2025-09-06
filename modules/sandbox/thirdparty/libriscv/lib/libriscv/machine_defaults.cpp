/**
 * Some default implementations of OS-specific I/O routines
 * stdout: Used by write/writev system calls
 * stdin:  Used by read/readv system calls
 * rdtime: Used by the RDTIME/RDTIMEH instructions
**/
#include "machine.hpp"
#include "internal_common.hpp"

#include <chrono>
extern "C" {
#ifdef _WIN32
	int write(int fd, const void *buf, unsigned count);
#else
	ssize_t write(int fd, const void *buf, size_t count);
#endif
}

namespace riscv
{
	// Default: Stdout allowed
	template <int W>
	void Machine<W>::default_printer(const Machine<W>&, const char* buffer, size_t len) {
		std::ignore = ::write(1, buffer, len);
	}
	// Default: Stdin *NOT* allowed
	template <int W>
	long Machine<W>::default_stdin(const Machine<W>&, char* /*buffer*/, size_t /*len*/) {
		return 0;
	}

	// Default: RDTIME produces monotonic time with *microsecond*-granularity
	template <int W>
	uint64_t Machine<W>::default_rdtime(const Machine<W>& machine) {
#ifdef __wasm__
		return 0;
#else
		auto now = std::chrono::steady_clock::now();
		auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
		if (!(machine.has_file_descriptors() && machine.fds().proxy_mode))
			micros &= ANTI_FINGERPRINTING_MASK_MICROS();
		return micros;
#endif
	}

#ifndef __GNUG__ /* Workaround for GCC bug */
	INSTANTIATE_32_IF_ENABLED(Machine);
	INSTANTIATE_64_IF_ENABLED(Machine);
	INSTANTIATE_128_IF_ENABLED(Machine);
#endif
}
