#include "../machine.hpp"

#include "../internal_common.hpp"
#include "../threads.hpp"

//#define SYSCALL_VERBOSE 1
#ifdef SYSCALL_VERBOSE
#define SYSPRINT(fmt, ...) \
	{ char syspbuf[1024]; machine.print(syspbuf, \
		snprintf(syspbuf, sizeof(syspbuf), fmt, ##__VA_ARGS__)); }
static constexpr bool verbose_syscalls = true;
#else
#define SYSPRINT(fmt, ...) /* fmt */
static constexpr bool verbose_syscalls = false;
#endif

#include <fcntl.h>
#include <signal.h>
#undef sa_handler
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#if !defined(__OpenBSD__) && !defined(TARGET_OS_IPHONE)
#include <sys/random.h>
#endif
extern "C" int dup3(int oldfd, int newfd, int flags);
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/uio.h>
#if __has_include(<termios.h>)
#include <termios.h>
#endif
#include <sys/syscall.h>
#ifndef EBADFD
#define EBADFD EBADF  // OpenBSD, FreeBSD
#endif
#define LINUX_SA_ONSTACK	0x08000000

namespace riscv {
template <int W>
extern void add_socket_syscalls(Machine<W>&);

template <int W>
struct guest_iovec {
	address_type<W> iov_base;
	address_type<W> iov_len;
};

#if defined(__APPLE__)
#include <mach/mach_time.h>
static int get_time(int clkid, struct timespec* ts) {
	if (clkid == CLOCK_REALTIME) {
		struct timeval tv;
		gettimeofday(&tv, NULL);
		ts->tv_sec = tv.tv_sec;
		ts->tv_nsec = tv.tv_usec * 1000;
		return 0;
	} else if (clkid == CLOCK_MONOTONIC) {
		uint64_t time = mach_absolute_time();
		mach_timebase_info_data_t timebase;
		mach_timebase_info(&timebase);
		double nsec = ((double)time * (double)timebase.numer)/((double)timebase.denom);
		ts->tv_sec = nsec * 1e-9;  
		ts->tv_nsec = nsec - (ts->tv_sec * 1e9);
		return 0;
	} else {
		return -1;
	}
}
#else
static int get_time(int clkid, struct timespec *ts)
{
	return clock_gettime(clkid, ts);
}
#endif

template <int W>
static void syscall_stub_zero(Machine<W>& machine) {
	SYSPRINT("SYSCALL stubbed (zero): %d\n", (int)machine.cpu.reg(17));
	machine.set_result(0);
}

template <int W>
static void syscall_stub_nosys(Machine<W>& machine) {
	SYSPRINT("SYSCALL stubbed (nosys): %d\n", (int)machine.cpu.reg(17));
	machine.set_result(-ENOSYS);
}

template <int W>
static void syscall_exit(Machine<W>& machine)
{
	// Stop sets the max instruction counter to zero, allowing most
	// instruction loops to end. It is, however, not the only way
	// to exit a program. Tighter integrations with the library should
	// provide their own methods.
	machine.stop();
}

template <int W>
static void syscall_ebreak(riscv::Machine<W>& machine)
{
	printf("\n>>> EBREAK at %#lX\n", (long) machine.cpu.pc());
	throw MachineException(UNHANDLED_SYSCALL, "EBREAK instruction");
}

template <int W>
static void syscall_sigaltstack(Machine<W>& machine)
{
	const auto ss = machine.sysarg(0);
	const auto old_ss = machine.sysarg(1);
	SYSPRINT("SYSCALL sigaltstack, tid=%d ss: 0x%lX old_ss: 0x%lX\n",
		machine.gettid(), (long)ss, (long)old_ss);

	auto& stack = machine.signals().per_thread(machine.gettid()).stack;

	if (old_ss != 0x0) {
		machine.copy_to_guest(old_ss, &stack, sizeof(stack));
	}
	if (ss != 0x0) {
		machine.copy_from_guest(&stack, ss, sizeof(stack));

		SYSPRINT("<<< sigaltstack sp: 0x%lX flags: 0x%X size: 0x%lX\n",
			(long)stack.ss_sp, stack.ss_flags, (long)stack.ss_size);
	}

	machine.set_result(0);
}

template <int W>
static void syscall_sigaction(Machine<W>& machine)
{
	const int sig = machine.sysarg(0);
	const auto action = machine.sysarg(1);
	const auto old_action = machine.sysarg(2);
	SYSPRINT("SYSCALL sigaction, signal: %d, action: 0x%lX old_action: 0x%lX\n",
		sig, (long)action, (long)old_action);
	if (sig == 0) return;

	auto& sigact = machine.sigaction(sig);

	struct kernel_sigaction {
		address_type<W> sa_handler;
		address_type<W> sa_flags;
		address_type<W> sa_mask;
	} sa {};
	if (old_action != 0x0) {
		sa.sa_handler = sigact.handler & ~address_type<W>(0xF);
		sa.sa_flags   = (sigact.altstack ? LINUX_SA_ONSTACK : 0x0);
		sa.sa_mask    = sigact.mask;
		machine.copy_to_guest(old_action, &sa, sizeof(sa));
	}
	if (action != 0x0) {
		machine.copy_from_guest(&sa, action, sizeof(sa));
		sigact.handler  = sa.sa_handler;
		sigact.altstack = (sa.sa_flags & LINUX_SA_ONSTACK) != 0;
		sigact.mask     = sa.sa_mask;
		SYSPRINT("<<< sigaction %d handler: 0x%lX altstack: %d\n",
			sig, (long)sigact.handler, sigact.altstack);
	}

	machine.set_result(0);
}

template <int W>
void syscall_getdents64(Machine<W>& machine)
{
	const int fd = machine.template sysarg<int>(0);
	const auto g_dirp = machine.sysarg(1);
	const auto count = machine.template sysarg<int>(2);

	SYSPRINT("SYSCALL getdents64, fd: %d, dirp: 0x%lX, count: %d\n",
		fd, (long)g_dirp, count);
	(void)count;

	if (machine.has_file_descriptors() && machine.fds().proxy_mode) {
#if defined(__linux__) && defined(__LP64__)
		const int real_fd = machine.fds().translate(fd);

		char buffer[4096];
		const int res = syscall(SYS_getdents64, real_fd, buffer, sizeof(buffer));
		if (res > 0)
		{
			machine.copy_to_guest(g_dirp, buffer, res);
		}
		machine.set_result_or_error(res);
#else
		(void)fd; (void)g_dirp;
		machine.set_result(-ENOSYS);
#endif
	} else {
		machine.set_result(-EBADF);
	}
}

template <int W>
void syscall_lseek(Machine<W>& machine)
{
	const int fd      = machine.template sysarg<int>(0);
	const auto offset = machine.sysarg(1);
	const int whence  = machine.template sysarg<int>(2);
	SYSPRINT("SYSCALL lseek, fd: %d, offset: 0x%lX, whence: %d\n",
		fd, (long)offset, whence);

	if (machine.has_file_descriptors()) {
		const int real_fd = machine.fds().get(fd);
#ifndef __wasm__
		long res = lseek(real_fd, offset, whence);
#else
		long res = -ENOSYS;
#endif
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}
}
template <int W>
static void syscall_read(Machine<W>& machine)
{
	const int  vfd     = machine.template sysarg<int>(0);
	const auto address = machine.sysarg(1);
	const size_t len   = machine.sysarg(2);
	SYSPRINT("SYSCALL read, vfd: %d addr: 0x%lX, len: %zu\n",
		vfd, (long)address, len);
	// We have special stdin handling
	if (vfd == 0) {
		// Arbitrary maximum read length
		if (len > 1024 * 1024 * 16) {
			machine.set_result(-ENOMEM);
			return;
		}
		// TODO: We can use gather buffers here to avoid the copy
		auto buffer = std::unique_ptr<char[]> (new char[len]);
		long result = machine.stdin_read(buffer.get(), len);
		if (result > 0) {
			machine.copy_to_guest(address, buffer.get(), result);
		}
		machine.set_result_or_error(result);
		return;
	} else if (machine.has_file_descriptors()) {
		const int real_fd = machine.fds().translate(vfd);

		std::array<riscv::vBuffer, 512> buffers;
		size_t cnt =
			machine.memory.gather_writable_buffers_from_range(buffers.size(), buffers.data(), address, len);
		const ssize_t res =
			readv(real_fd, (const iovec *)&buffers[0], cnt);
		machine.set_result_or_error(res);
		SYSPRINT("SYSCALL read, fd: %d from vfd: %d = %ld\n",
				 real_fd, vfd, (long)machine.return_value());
	} else {
		machine.set_result(-EBADF);
		SYSPRINT("SYSCALL read, vfd: %d = -EBADF\n", vfd);
	}
}
template <int W>
static void syscall_pread64(Machine<W>& machine)
{
	const int  vfd     = machine.template sysarg<int>(0);
	const auto address = machine.sysarg(1);
	const size_t len   = machine.sysarg(2);
	const auto offset  = machine.sysarg(3);
	SYSPRINT("SYSCALL pread64, vfd: %d addr: 0x%lX, len: %zu, offset: %lu\n",
		vfd, (long)address, len, (long)offset);
	if (machine.has_file_descriptors()) {
		const int real_fd = machine.fds().translate(vfd);

		std::array<riscv::vBuffer, 512> buffers;
		const size_t cnt =
			machine.memory.gather_writable_buffers_from_range(buffers.size(), buffers.data(), address, len);
#if defined(__linux__) && defined(SYS_preadv)
		const ssize_t res =
			syscall(SYS_preadv, real_fd, (const iovec *)&buffers[0], cnt, offset);
#elif defined(__wasm__)
		const ssize_t res = -ENOSYS;
#else
		size_t total = 0;
		ssize_t res = 0;
		for (size_t i = 0; i < cnt; i++) {
			res = pread(real_fd, buffers[i].ptr, buffers[i].len, offset + total);
			if (res < 0)
				break;
			total += res;
			if ((size_t)res < buffers[i].len) {
				res = total; // Return the total read
				break;
			}
		}
#endif
		machine.set_result_or_error(res);
		SYSPRINT("SYSCALL pread64, fd: %d from vfd: %d => %ld\n",
				 real_fd, vfd, (long)machine.return_value());
	} else {
		machine.set_result(-EBADF);
		SYSPRINT("SYSCALL pread64, vfd: %d => -EBADF\n", vfd);
	}
}
template <int W>
static void syscall_write(Machine<W>& machine)
{
	const int  vfd     = machine.template sysarg<int>(0);
	const auto address = machine.sysarg(1);
	const size_t len   = machine.sysarg(2);
	SYSPRINT("SYSCALL write, fd: %d addr: 0x%lX, len: %zu\n",
		vfd, (long)address, len);
	// Zero-copy retrieval of buffers
	std::array<riscv::vBuffer, 64> buffers;

	if (vfd == 1 || vfd == 2) {
		size_t cnt =
			machine.memory.gather_buffers_from_range(buffers.size(), buffers.data(), address, len);
		for (size_t i = 0; i < cnt; i++) {
			machine.print(buffers[i].ptr, buffers[i].len);
		}
		machine.set_result(len);
	} else if (machine.has_file_descriptors() && machine.fds().permit_write(vfd)) {
		int real_fd = machine.fds().translate(vfd);
		size_t cnt =
			machine.memory.gather_buffers_from_range(buffers.size(), buffers.data(), address, len);
		const ssize_t res =
			writev(real_fd, (struct iovec *)&buffers[0], cnt);
		SYSPRINT("SYSCALL write(real fd: %d iovec: %zu) = %ld\n",
			real_fd, cnt, res);
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}
}

template <int W>
static void syscall_readv(Machine<W>& machine)
{
	const int  vfd    = machine.template sysarg<int>(0);
	const auto iov_g  = machine.sysarg(1);
	const auto count  = machine.template sysarg<int>(2);
	if (count < 1 || count > 128) {
		machine.set_result(-EINVAL);
		return;
	}

	int real_fd = -1;
	if (vfd == 1 || vfd == 2) {
		real_fd = -1;
	} else if (machine.has_file_descriptors()) {
		real_fd = machine.fds().translate(vfd);
	}

	if (real_fd < 0) {
		machine.set_result(-EBADF);
	} else {
		const size_t iov_size = sizeof(guest_iovec<W>) * count;

		// Retrieve the guest IO vec
		std::array<guest_iovec<W>, 128> g_vec;
		machine.copy_from_guest(g_vec.data(), iov_g, iov_size);

		// Convert each iovec buffer to host buffers
		std::array<riscv::vBuffer, 64> buffers;
		size_t vec_cnt = 0;

		for (int i = 0; i < count; i++) {
			// The host buffers come directly from guest memory
			vec_cnt += machine.memory.gather_writable_buffers_from_range(
				buffers.size() - vec_cnt, &buffers[vec_cnt], g_vec[i].iov_base, g_vec[i].iov_len);
		}

		const ssize_t res = readv(real_fd, (struct iovec *)&buffers[0], vec_cnt);
		machine.set_result_or_error(res);
	}
	SYSPRINT("SYSCALL readv(vfd: %d iov: 0x%lX cnt: %d) = %ld\n",
		vfd, (long)iov_g, count, (long)machine.return_value());
} // readv

template <int W>
static void syscall_writev(Machine<W>& machine)
{
	const int  vfd    = machine.template sysarg<int>(0);
	const auto iov_g  = machine.sysarg(1);
	const auto count  = machine.template sysarg<int>(2);
	if constexpr (verbose_syscalls) {
		printf("SYSCALL writev, iov: 0x%lX  cnt: %d\n", (long)iov_g, count);
	}
	if (count < 0 || count > 256) {
		machine.set_result(-EINVAL);
		return;
	}

	int real_fd = -1;
	if (vfd == 1 || vfd == 2) {
		real_fd = vfd;
	} else if (machine.has_file_descriptors()) {
		real_fd = machine.fds().translate(vfd);
	}

	if (real_fd < 0) {
		machine.set_result(-EBADF);
	} else {
		std::array<guest_iovec<W>, 256> vec;
		machine.memory.memcpy_out(vec.data(), iov_g, sizeof(guest_iovec<W>) * count);

		/* Zero-copy retrieval of buffers */
		std::array<riscv::vBuffer, 64> buffers;
		size_t vec_cnt = 0;

		for (int i = 0; i < count; i++)
		{
			auto& iov = vec.at(i);
			auto src_g = (address_type<W>) iov.iov_base;
			auto len_g = (size_t) iov.iov_len;

			vec_cnt +=
				machine.memory.gather_buffers_from_range(buffers.size() - vec_cnt, &buffers[vec_cnt], src_g, len_g);
		}

		ssize_t res = 0;
		if (real_fd == 1 || real_fd == 2) {
			// STDOUT, STDERR
			for (size_t i = 0; i < vec_cnt; i++) {
				machine.print(buffers[i].ptr, buffers[i].len);
				res += buffers[i].len;
			}
		} else {
			// General file descriptor
			res = writev(real_fd, (const struct iovec *)buffers.data(), vec_cnt);
		}
		machine.set_result_or_error(res);
	}
	if constexpr (verbose_syscalls) {
		printf("SYSCALL writev, vfd: %d real_fd: %d -> %ld\n",
			vfd, real_fd, long(machine.return_value()));
	}
} // writev

template <int W>
static void syscall_openat(Machine<W>& machine)
{
	const int dir_fd = machine.template sysarg<int>(0);
	const auto g_path = machine.sysarg(1);
	const int flags  = machine.template sysarg<int>(2);
	// We do it this way to prevent accessing memory out of bounds
	std::string path = machine.memory.memstring(g_path);

	SYSPRINT("SYSCALL openat, dir_fd: %d path: %s flags: %X\n",
		dir_fd, path.c_str(), flags);

	if (machine.has_file_descriptors() && machine.fds().permit_filesystem) {

		if (machine.fds().filter_open != nullptr) {
			// filter_open() can modify the path
			if (!machine.fds().filter_open(machine.template get_userdata<void>(), path)) {
				machine.set_result(-EPERM);
				SYSPRINT("SYSCALL openat(path: %s) => %d\n",
					path.c_str(), machine.template return_value<int>());
				return;
			}
		}
		int real_fd = openat(machine.fds().translate(dir_fd), path.c_str(), flags);
		if (real_fd > 0) {
			const int vfd = machine.fds().assign_file(real_fd);
			machine.set_result(vfd);
		} else {
			// Translate errno() into kernel API return value
			machine.set_result(-errno);
		}
		SYSPRINT("SYSCALL openat(real_fd: %d) => %d\n",
			real_fd, machine.template return_value<int>());
		return;
	}

	machine.set_result(-EBADF);
	SYSPRINT("SYSCALL openat => %d\n", machine.template return_value<int>());
}

template <int W>
static void syscall_close(riscv::Machine<W>& machine)
{
	const int vfd = machine.template sysarg<int>(0);

	if (vfd >= 0 && vfd <= 2) {
		// TODO: Do we really want to close them?
		machine.set_result(0);
	} else if (machine.has_file_descriptors()) {
		const int res = machine.fds().erase(vfd);
		if (res > 0) {
			::close(res);
		}
		machine.set_result(res >= 0 ? 0 : -EBADF);
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL close(vfd: %d) => %d\n",
		vfd, machine.template return_value<int>());
}

template <int W>
static void syscall_dup(Machine<W>& machine)
{
	const int vfd = machine.template sysarg<int>(0);
	SYSPRINT("SYSCALL dup, fd: %d\n", vfd);

	if (machine.has_file_descriptors()) {
		int real_fd = machine.fds().translate(vfd);
		int res = dup(real_fd);
		machine.set_result_or_error(res);
		return;
	}
	machine.set_result(-EBADF);
}

template <int W>
static void syscall_dup3(Machine<W>& machine)
{
	const int old_vfd = machine.template sysarg<int>(0);
	const int new_vfd = machine.template sysarg<int>(1);
	const int flags = machine.template sysarg<int>(2);

	if (machine.has_file_descriptors()) {
		int real_old_fd = machine.fds().translate(old_vfd);
		int real_new_fd = machine.fds().translate(new_vfd);
#if defined(__linux__) || defined(__FreeBSD__) || defined(__OpenBSD__)
		int res = dup3(real_old_fd, real_new_fd, flags);
		if (res > 0) {
			res = machine.fds().assign_file(res);
		}
		machine.set_result_or_error(res);
#else
		(void)flags;
		(void)real_old_fd;
		(void)real_new_fd;
		machine.set_result(-ENOSYS);
#endif
	} else {
		machine.set_result(-EBADF);
	}

	SYSPRINT("SYSCALL dup3(old_vfd: %d, new_vfd: %d, flags: 0x%X) => %d\n",
		old_vfd, new_vfd, flags, (int)machine.return_value());
}

int create_pipe(int* pipes, int flags) {
	#if defined(__APPLE__)
		// On macOS, we don't have pipe2, so we need to use pipe and then set the flags manually.
		int res = pipe(pipes);
		if (res == 0 && flags != 0) {
			if (flags & O_CLOEXEC) {
				fcntl(pipes[0], F_SETFD, FD_CLOEXEC);
				fcntl(pipes[1], F_SETFD, FD_CLOEXEC);
			}
			if (flags & O_NONBLOCK) {
				fcntl(pipes[0], F_SETFL, O_NONBLOCK);
				fcntl(pipes[1], F_SETFL, O_NONBLOCK);
			}
		}
		return res;
	#else
		return pipe2(pipes, flags);
	#endif
}

template <int W>
static void syscall_pipe2(Machine<W>& machine)
{
	const auto vfd_array = machine.sysarg(0);
	const auto flags = machine.template sysarg<int>(1);

	if (machine.has_file_descriptors()) {
		int pipes[2];
		int res = create_pipe(pipes, flags);
		if (res == 0) {
			int vpipes[2];
			vpipes[0] = machine.fds().assign_file(pipes[0]);
			vpipes[1] = machine.fds().assign_file(pipes[1]);
			machine.copy_to_guest(vfd_array, vpipes, sizeof(vpipes));
			machine.set_result(0);
		} else {
			machine.set_result_or_error(res);
		}
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL pipe2, fd array: 0x%lX flags: %d = %ld\n",
		(long)vfd_array, flags, (long)machine.return_value());
}

template <int W>
static void syscall_fcntl(Machine<W>& machine)
{
	const int vfd = machine.template sysarg<int>(0);
	const auto cmd = machine.template sysarg<int>(1);
	const auto arg1 = machine.sysarg(2);
	const auto arg2 = machine.sysarg(3);
	const auto arg3 = machine.sysarg(4);
	int real_fd = -EBADFD;

	if (machine.has_file_descriptors()) {
		real_fd = machine.fds().translate(vfd);
		int res = fcntl(real_fd, cmd, arg1, arg2, arg3);
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL fcntl, fd: %d (real_fd: %d)  cmd: 0x%X arg1: 0x%lX => %d\n",
		vfd, real_fd, cmd, (long)arg1, (int)machine.return_value());
}

template <int W>
static void syscall_ioctl(Machine<W>& machine)
{
	const int vfd = machine.template sysarg<int>(0);
	const auto req = machine.template sysarg<uint64_t>(1);
	const auto arg1 = machine.sysarg(2);
	const auto arg2 = machine.sysarg(3);
	const auto arg3 = machine.sysarg(4);
	const auto arg4 = machine.sysarg(5);
	SYSPRINT("SYSCALL ioctl, fd: %d  req: 0x%lX\n", vfd, req);

	if (machine.has_file_descriptors()) {
		if (machine.fds().filter_ioctl != nullptr) {
			if (!machine.fds().filter_ioctl(machine.template get_userdata<void>(), req)) {
				machine.set_result(-EPERM);
				return;
			}
		}

		int real_fd = machine.fds().translate(vfd);

#if __has_include(<termios.h>) && defined(TCGETS)
		// Terminal control - ~unsandboxable, but we permit with proxy mode
		if (machine.fds().proxy_mode && (req == TCGETS || req == TCSETS || req == TCSETSW || req == TCSETSF)) {
			struct termios term {};
			machine.copy_from_guest(&term, arg1, sizeof(term));
			int res = ioctl(real_fd, req, &term);
			if (req == TCGETS)
				machine.copy_to_guest(arg1, &term, sizeof(term));
			machine.set_result_or_error(res);
			return;
		}
#endif

		if (machine.fds().proxy_mode && req == FIONBIO) {
			int opt = machine.memory.template read<int>(arg1);
			int res = ioctl(real_fd, req, &opt);
			machine.set_result_or_error(res);
			return;
		}

		int res = ioctl(real_fd, req, arg1, arg2, arg3, arg4);
		machine.set_result_or_error(res);
		return;
	}
	machine.set_result(-EBADF);
}

template <int W>
void syscall_readlinkat(Machine<W>& machine)
{
	const int vfd = machine.template sysarg<int>(0);
	const auto g_path = machine.sysarg(1);
	const auto g_buf = machine.sysarg(2);
	const auto bufsize = machine.sysarg(3);

	const std::string original_path = machine.memory.memstring(g_path);

	SYSPRINT("SYSCALL readlinkat, fd: %d path: %s buffer: 0x%lX size: %zu\n",
		vfd, original_path.c_str(), (long)g_buf, (size_t)bufsize);

	char buffer[1024];
	if (bufsize > sizeof(buffer)) {
		machine.set_result(-ENOMEM);
		return;
	}

	if (machine.has_file_descriptors()) {

		if (machine.fds().filter_readlink != nullptr) {
			std::string path = original_path;
			if (!machine.fds().filter_readlink(machine.template get_userdata<void>(), path)) {
				machine.set_result(-EPERM);
				return;
			}
			// Readlink always rewrites the answer
			machine.copy_to_guest(g_buf, path.c_str(), path.size());
			machine.set_result(path.size());
			return;
		}
		const int real_fd = machine.fds().translate(vfd);

		const int res = readlinkat(real_fd, original_path.c_str(), buffer, bufsize);
		if (res > 0) {
			// TODO: Only necessary if g_buf is not sequential.
			machine.copy_to_guest(g_buf, buffer, res);
		}

		machine.set_result_or_error(res);
		SYSPRINT("SYSCALL readlinkat, fd: %d path: %s buffer: 0x%lX size: %zu => %d\n",
			vfd, original_path.c_str(), (long)g_buf, (size_t)bufsize, (int)machine.return_value());
		return;
	}

	machine.set_result(-ENOSYS);
}

// The RISC-V stat structure is different from x86
struct riscv_stat {
	uint64_t st_dev;		/* Device.  */
	uint64_t st_ino;		/* File serial number.  */
	uint32_t st_mode;	/* File mode.  */
	uint32_t st_nlink;	/* Link count.  */
	uint32_t st_uid;		/* User ID of the file's owner.  */
	uint32_t st_gid;		/* Group ID of the file's group. */
	uint64_t st_rdev;	/* Device number, if device.  */
	uint64_t __pad1;
	int64_t  st_size;	/* Size of file, in bytes.  */
	int32_t  st_blksize;	/* Optimal block size for I/O.  */
	int32_t  __pad2;
	int64_t  st_blocks;	/* Number 512-byte blocks allocated. */
	// TODO: The following are all 32-bit on 32-bit RISC-V
	int64_t  rv_atime;	/* Time of last access.  */
	uint64_t rv_atime_nsec;
	int64_t  rv_mtime;	/* Time of last modification.  */
	uint64_t rv_mtime_nsec;
	int64_t  rv_ctime;	/* Time of last status change.  */
	uint64_t rv_ctime_nsec;
	uint32_t __unused4;
	uint32_t __unused5;
};
inline void copy_stat_buffer(struct stat& st, struct riscv_stat& rst)
{
	rst.st_dev = st.st_dev;
	rst.st_ino = st.st_ino;
	rst.st_mode = st.st_mode;
	rst.st_nlink = st.st_nlink;
	rst.st_uid = st.st_uid;
	rst.st_gid = st.st_gid;
	rst.st_rdev = st.st_rdev;
	rst.st_size = st.st_size;
	rst.st_blksize = st.st_blksize;
	rst.st_blocks = st.st_blocks;
	rst.rv_atime = st.st_atime;
	#ifdef __APPLE__
	rst.rv_atime_nsec = st.st_atimespec.tv_nsec;
	#else
	#ifdef __USE_MISC
	rst.rv_atime_nsec = st.st_atim.tv_nsec;
	#else
	rst.rv_atime_nsec = 0; // or another appropriate value
	#endif
	#endif
	rst.rv_mtime = st.st_mtime;
	#ifdef __APPLE__
	rst.rv_mtime_nsec = st.st_mtimespec.tv_nsec;
	#else
	#ifdef __USE_MISC
	rst.rv_mtime_nsec = st.st_mtim.tv_nsec;
	#else
	rst.rv_mtime_nsec = 0; // or another appropriate value
	#endif
	#endif
	rst.rv_ctime = st.st_ctime;
	#ifdef __APPLE__
	rst.rv_ctime_nsec = st.st_ctimespec.tv_nsec;
	#else
	#ifdef __USE_MISC
	rst.rv_ctime_nsec = st.st_ctim.tv_nsec;
	#else
	rst.rv_ctime_nsec = 0; // or another appropriate value
	#endif
	#endif
}


template <int W>
static void syscall_getcwd(Machine<W>& machine)
{
	const auto g_buf = machine.sysarg(0);
	[[maybe_unused]] const auto size = machine.sysarg(1);

	auto& cwd = machine.fds().cwd;
	if (!cwd.empty()) {
		machine.copy_to_guest(g_buf, cwd.c_str(), cwd.size()+1);
		machine.set_result(cwd.size()+1);
	} else {
		machine.set_result(-1);
	}
	SYSPRINT("SYSCALL getcwd, buffer: 0x%lX size: %ld => %ld\n",
		(long)g_buf, (long)size, (long)machine.return_value());
}

template <int W>
static void syscall_fstatat(Machine<W>& machine)
{
	const auto vfd = machine.template sysarg<int> (0);
	const auto g_path = machine.sysarg(1);
	const auto g_buf = machine.sysarg(2);
	const auto flags = machine.template sysarg<int> (3);

	std::string path = machine.memory.memstring(g_path);

	if (machine.has_file_descriptors()) {

		int real_fd = machine.fds().translate(vfd);

		if (machine.fds().filter_stat != nullptr && !path.empty()) {
			if (!machine.fds().filter_stat(machine.template get_userdata<void>(), path)) {
				machine.set_result(-EPERM);
				return;
			}
		}

		struct stat st;
		int res = -1;
		if (!path.empty())
			res = ::fstatat(real_fd, path.c_str(), &st, flags);
		else
			res = ::fstat(real_fd, &st);
		if (res == 0) {
#ifndef __riscv
			// Convert to RISC-V structure
			struct riscv_stat rst;
			copy_stat_buffer(st, rst);
			machine.copy_to_guest(g_buf, &rst, sizeof(rst));
#else // on RISC-V no conversion
			machine.copy_to_guest(g_buf, &st, sizeof(st));
#endif
		}
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-ENOSYS);
	}
	SYSPRINT("SYSCALL fstatat, fd: %d path: %s buf: 0x%lX flags: %#x) => %d\n",
			vfd, path.c_str(), (long)g_buf, flags, (int)machine.return_value());
}

template <int W>
static void syscall_faccessat(Machine<W>& machine)
{
	const auto fd = AT_FDCWD;
	const auto g_path = machine.sysarg(1);
	const auto mode   = machine.template sysarg<int>(2);
	const auto flags  = machine.template sysarg<int>(3);

	const auto path = machine.memory.memstring(g_path);

	SYSPRINT("SYSCALL faccessat, fd: %d path: %s)\n",
			fd, path.c_str());

	const int res =
		faccessat(fd, path.c_str(), mode, flags);
	machine.set_result_or_error(res);
}

template <int W>
static void syscall_fstat(Machine<W>& machine)
{
	const auto vfd = machine.template sysarg<int> (0);
	const auto g_buf = machine.sysarg(1);

	if (machine.has_file_descriptors()) {

		const int real_fd = machine.fds().translate(vfd);

		struct stat st;
		const int res = ::fstat(real_fd, &st);
		if (res == 0) {
#ifndef __riscv
			// Convert to RISC-V structure
			struct riscv_stat rst;
			std::memset(&rst, 0, sizeof(rst));
			copy_stat_buffer(st, rst);
			machine.copy_to_guest(g_buf, &rst, sizeof(rst));
#else // on RISC-V no conversion
			machine.copy_to_guest(g_buf, &st, sizeof(st));
#endif
		}
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-ENOSYS);
	}

	SYSPRINT("SYSCALL fstat, fd: %d buf: 0x%lX) => %d\n",
			 vfd, (long)g_buf, (int)machine.return_value());
}

template <int W>
static void syscall_gettimeofday(Machine<W>& machine)
{
	const auto buffer = machine.sysarg(0);
	SYSPRINT("SYSCALL gettimeofday, buffer: 0x%lX\n", (long)buffer);
	struct timeval tv;
	const int res = gettimeofday(&tv, nullptr);
	if (res >= 0) {
		if (!(machine.has_file_descriptors() && machine.fds().proxy_mode))
			tv.tv_usec &= ANTI_FINGERPRINTING_MASK_MICROS();
		machine.copy_to_guest(buffer, &tv, sizeof(tv));
	}
	machine.set_result_or_error(res);
}
template <int W>
static void syscall_clock_gettime(Machine<W>& machine)
{
	const auto clkid = machine.template sysarg<int>(0);
	const auto buffer = machine.sysarg(1);
	SYSPRINT("SYSCALL clock_gettime, clkid: %x buffer: 0x%lX\n",
		clkid, (long)buffer);

	struct timespec ts;
	const int res = get_time(clkid, &ts);
	if (res >= 0) {
		if (!(machine.has_file_descriptors() && machine.fds().proxy_mode))
			ts.tv_nsec &= ANTI_FINGERPRINTING_MASK_NANOS();
		if constexpr (W == 4) {
			int32_t ts32[2] = {(int) ts.tv_sec, (int) ts.tv_nsec};
			machine.copy_to_guest(buffer, &ts32, sizeof(ts32));
		} else {
			machine.copy_to_guest(buffer, &ts, sizeof(ts));
		}
	}
	machine.set_result_or_error(res);
}
template <int W>
static void syscall_clock_gettime64(Machine<W>& machine)
{
	const auto clkid = machine.template sysarg<int>(0);
	const auto buffer = machine.sysarg(1);
	SYSPRINT("SYSCALL clock_gettime64, clkid: %x buffer: 0x%lX\n",
		clkid, (long)buffer);

	struct timespec ts;
	int res = get_time(clkid, &ts);

	if (res >= 0) {
		if (!(machine.has_file_descriptors() && machine.fds().proxy_mode))
			ts.tv_nsec &= ANTI_FINGERPRINTING_MASK_NANOS();
		struct {
			int64_t tv_sec;
			int64_t tv_nsec;
		} kernel_ts;
		kernel_ts.tv_sec  = ts.tv_sec;
		kernel_ts.tv_nsec = ts.tv_nsec;
		machine.copy_to_guest(buffer, &kernel_ts, sizeof(kernel_ts));
	}
	machine.set_result_or_error(res);
}
template <int W>
static void syscall_nanosleep(Machine<W>& machine)
{
	const auto g_req = machine.sysarg(0);
	const auto g_rem = machine.sysarg(1);
	SYSPRINT("SYSCALL nanosleep, req: 0x%lX rem: 0x%lX\n",
		(long)g_req, (long)g_rem);

	struct timespec ts_req;
	machine.copy_from_guest(&ts_req, g_req, sizeof(ts_req));
	if (!(machine.has_file_descriptors() && machine.fds().proxy_mode))
		ts_req.tv_nsec &= ANTI_FINGERPRINTING_MASK_NANOS();

	struct timespec ts_rem;
	if (g_rem)
		machine.copy_from_guest(&ts_rem, g_rem, sizeof(ts_rem));

	const int res = nanosleep(&ts_req, g_rem != 0x0 ? &ts_rem : nullptr);
	if (res >= 0) {
		machine.copy_to_guest(g_req, &ts_req, sizeof(ts_req));
		if (g_rem)
			machine.copy_to_guest(g_rem, &ts_rem, sizeof(ts_rem));
	}
	machine.set_result_or_error(res);
}
template <int W>
static void syscall_clock_nanosleep(Machine<W>& machine)
{
	const auto g_request = machine.sysarg(2);
	const auto g_remain = machine.sysarg(3);

	struct timespec ts_req;
	struct timespec ts_rem;
	machine.copy_from_guest(&ts_req, g_request, sizeof(ts_req));
	if (!(machine.has_file_descriptors() && machine.fds().proxy_mode))
		ts_req.tv_nsec &= ANTI_FINGERPRINTING_MASK_NANOS();

	const int res = nanosleep(&ts_req, &ts_rem);
	if (res >= 0 && g_remain != 0x0) {
		machine.copy_to_guest(g_remain, &ts_rem, sizeof(ts_rem));
	}
	machine.set_result_or_error(res);

	SYSPRINT("SYSCALL clock_nanosleep, req: 0x%lX rem: 0x%lX = %ld\n",
		(long)g_request, (long)g_remain, (long)machine.return_value());
}

template <int W>
static void syscall_uname(Machine<W>& machine)
{
	const auto buffer = machine.sysarg(0);
	SYSPRINT("SYSCALL uname, buffer: 0x%lX\n", (long)buffer);
	static constexpr int UTSLEN = 65;
	struct {
		char sysname [UTSLEN];
		char nodename[UTSLEN];
		char release [UTSLEN];
		char version [UTSLEN];
		char machine [UTSLEN];
		char domain  [UTSLEN];
	} uts;
	strcpy(uts.sysname, "RISC-V C++ Emulator");
	strcpy(uts.nodename,"libriscv");
	strcpy(uts.release, "5.6.0");
	strcpy(uts.version, "");
	if constexpr (W == 4)
		strcpy(uts.machine, "rv32imafdc");
	else if constexpr (W == 8)
		strcpy(uts.machine, "rv64imafdc");
	else
		strcpy(uts.machine, "rv128imafdc");
	strcpy(uts.domain,  "(none)");

	machine.copy_to_guest(buffer, &uts, sizeof(uts));
	machine.set_result(0);
}

template <int W>
static void syscall_capget(Machine<W>& machine)
{
	const auto header_ptr = machine.sysarg(0);
	const auto data_ptr = machine.sysarg(1);

	struct __user_cap_header_struct {
		uint32_t version;
		int pid;
	};

	struct __user_cap_data_struct {
		uint32_t effective;
		uint32_t permitted;
		uint32_t inheritable;
	};

	__user_cap_header_struct header;
	__user_cap_data_struct data;

	// Copy the header from guest to host
	machine.copy_from_guest(&header, header_ptr, sizeof(header));

	// Initialize the header structure
	if (header.version != 0x20080522) {
		// Unsupported version, set error
		machine.set_result_or_error(-EINVAL);
	} else {
		// Here you would typically interact with the capability subsystem of the
		// emulated environment to get the actual capabilities.
		// For simplicity, let's assume no capabilities:
		data.effective = 0;
		data.permitted = 0;
		data.inheritable = 0;

		// Copy the data back to the guest
		machine.copy_to_guest(data_ptr, &data, sizeof(data));

		// Set result to 0 indicating success
		machine.set_result_or_error(0);
	}

	SYSPRINT("SYSCALL capget, header: 0x%lX, data: 0x%lX => %ld\n",
			 (long)header_ptr, (long)data_ptr, (long)machine.return_value());
}

template <int W>
static void syscall_brk(Machine<W>& machine)
{
	auto new_end = machine.sysarg(0);
	if (new_end > machine.memory.heap_address() + Memory<W>::BRK_MAX) {
		new_end = machine.memory.heap_address() + Memory<W>::BRK_MAX;
	} else if (new_end < machine.memory.heap_address()) {
		new_end = machine.memory.heap_address();
	}

	if constexpr (verbose_syscalls) {
		printf("SYSCALL brk, new_end: 0x%lX  mmap_start: 0x%lX\n",
			(long)new_end, (long)machine.memory.mmap_start());
	}
	machine.set_result(new_end);
}

#if defined(__APPLE__)
	#include <Security/Security.h>
#endif

template <int W>
static void syscall_getrandom(Machine<W>& machine)
{
	const auto g_addr = machine.sysarg(0);
	const auto g_len  = machine.sysarg(1);

	char buffer[256];
	if (g_len > sizeof(buffer)) {
		machine.set_result(-1);
		return;
	}
	const size_t need = std::min((size_t)g_len, sizeof(buffer));
#if defined(__OpenBSD__)
	const ssize_t result = need; // always success
	arc4random_buf(buffer, need);
#elif defined(__APPLE__)
	#if TARGET_OS_IPHONE
	const ssize_t result = need;
	#else
	const int sec_result = SecRandomCopyBytes(kSecRandomDefault, need, (uint8_t *)buffer);
	const ssize_t result = (sec_result == errSecSuccess) ? need : -1;
	#endif
#elif defined(__ANDROID__) || defined(__wasm__)
	for (size_t i = 0; i < need; ++i) {
		buffer[i] ^= rand() & 0xFF; // XXX: Not secure
	}
	const ssize_t result = need;
#else
	const ssize_t result = getrandom(buffer, need, 0);
#endif
	if (result > 0) {
		machine.copy_to_guest(g_addr, buffer, result);
		// getrandom() is a slow syscall, penalize it
		machine.penalize(20'000 * result); // 20K insn per byte
	}
	machine.set_result(result);

	if constexpr (verbose_syscalls) {
		printf("SYSCALL getrandom(addr=0x%lX, len=%ld) = %ld\n",
			(long)g_addr, (long)g_len, (long)machine.return_value());
	}
}

#if defined(__linux__) && !defined(__ANDROID__)
template <int W>
static void syscall_statx(Machine<W>& machine)
{
	machine.set_result(-ENOSYS);
	return;

	const int   dir_fd = machine.template sysarg<int> (0);
	const auto  g_path = machine.sysarg(1);
	const int    flags = machine.template sysarg<int> (2);
	const auto    mask = machine.template sysarg<uint32_t> (3);
	const auto  buffer = machine.sysarg(4);

	const auto path = machine.memory.memstring(g_path);

	SYSPRINT("SYSCALL statx, fd: %d path: %s flags: %x buf: 0x%lX)\n",
			dir_fd, path.c_str(), flags, (long)buffer);

	if (machine.has_file_descriptors() && machine.fds().proxy_mode) {
		if (machine.fds().filter_stat != nullptr) {
			if (!machine.fds().filter_stat(machine.template get_userdata<void>(), path)) {
				machine.set_result(-EPERM);
				return;
			}
		}

		struct statx st;
		int res = ::statx(dir_fd, path.c_str(), flags, mask, &st);
		if (res == 0) {
			machine.copy_to_guest(buffer, &st, sizeof(struct statx));
		}
		machine.set_result_or_error(res);
		return;
	}
	machine.set_result(-ENOSYS);
}
#endif // __linux__

#include "syscalls_mman.cpp"

#include "syscalls_select.cpp"
#include "syscalls_poll.cpp"
#ifdef __linux__
#include "syscalls_epoll.cpp"
#else
#include "../win32/epoll.cpp"
#endif

template <int W>
void Machine<W>::setup_newlib_syscalls()
{
	install_syscall_handler(57, syscall_stub_zero<W>); // close
	install_syscall_handler(62, syscall_lseek<W>);
	install_syscall_handler(63, syscall_read<W>);
	install_syscall_handler(64, syscall_write<W>);
	install_syscall_handler(80, syscall_stub_nosys<W>); // fstat
	install_syscall_handler(93, syscall_exit<W>);
	install_syscall_handler(169, syscall_gettimeofday<W>);
	install_syscall_handler(214, syscall_brk<W>);
}
template <int W>
void Machine<W>::setup_newlib_syscalls(bool filesystem)
{
	setup_newlib_syscalls();

	if (filesystem)
		m_fds.reset(new FileDescriptors);
}

template <int W>
void Machine<W>::setup_linux_syscalls(bool filesystem, bool sockets)
{
	install_syscall_handler(SYSCALL_EBREAK, syscall_ebreak<W>);

	// getcwd
	install_syscall_handler(17, syscall_getcwd<W>);

	// eventfd2
	install_syscall_handler(19, syscall_eventfd2<W>);
	// epoll_create
	install_syscall_handler(20, syscall_epoll_create<W>);
	// epoll_ctl
	install_syscall_handler(21, syscall_epoll_ctl<W>);
	// epoll_pwait
	install_syscall_handler(22, syscall_epoll_pwait<W>);
	// dup
	install_syscall_handler(23, syscall_dup<W>);
	// dup3
	install_syscall_handler(24, syscall_dup3<W>);
	// fcntl
	install_syscall_handler(25, syscall_fcntl<W>);
	// ioctl
	install_syscall_handler(29, syscall_ioctl<W>);
	// faccessat
	install_syscall_handler(48, syscall_faccessat<W>);

	install_syscall_handler(56, syscall_openat<W>);
	install_syscall_handler(57, syscall_close<W>);
	install_syscall_handler(59, syscall_pipe2<W>);
	install_syscall_handler(61, syscall_getdents64<W>);
	install_syscall_handler(62, syscall_lseek<W>);
	install_syscall_handler(63, syscall_read<W>);
	install_syscall_handler(64, syscall_write<W>);
	install_syscall_handler(65, syscall_readv<W>);
	install_syscall_handler(66, syscall_writev<W>);
	install_syscall_handler(67, syscall_pread64<W>);
	install_syscall_handler(72, syscall_pselect<W>);
#ifdef __wasm__
	install_syscall_handler(73, syscall_stub_zero<W>);
	install_syscall_handler(78, syscall_stub_nosys<W>);
#else
	install_syscall_handler(73, syscall_ppoll<W>);
	install_syscall_handler(78, syscall_readlinkat<W>);
#endif
	// 79: fstatat
	install_syscall_handler(79, syscall_fstatat<W>);
	// 80: fstat
	install_syscall_handler(80, syscall_fstat<W>);
	// 90: capget
	install_syscall_handler(90, syscall_capget<W>);

	install_syscall_handler(93, syscall_exit<W>);
	// 94: exit_group (exit process)
	install_syscall_handler(94, syscall_exit<W>);

	// If no multi-threading has been initialized, we should install
	// stub syscall handlers for some multi-threading functionality
	if (!this->has_threads())
	{
		// set_tid_address
		install_syscall_handler(96, syscall_stub_zero<W>);
		// set_robust_list
		install_syscall_handler(99, syscall_stub_zero<W>);
		// prlimit64
		this->install_syscall_handler(261,
		[] (Machine<W>& machine) {
			const int resource = machine.template sysarg<int> (1);
			const auto old_addr = machine.template sysarg<address_type<W>> (3);
			struct {
				address_type<W> cur = 0;
				address_type<W> max = 0;
			} lim;
			constexpr int RISCV_RLIMIT_STACK = 3;
			if (old_addr != 0) {
				if (resource == RISCV_RLIMIT_STACK) {
					lim.cur = 0x200000;
					lim.max = 0x200000;
				}
				machine.copy_to_guest(old_addr, &lim, sizeof(lim));
				machine.set_result(0);
			} else {
				machine.set_result(-EINVAL);
			}
			SYSPRINT("SYSCALL prlimit64(...) = %d\n", machine.return_value<int>());
		});
	}

	// nanosleep
	install_syscall_handler(101, syscall_nanosleep<W>);
	// clock_gettime
	install_syscall_handler(113, syscall_clock_gettime<W>);
	install_syscall_handler(403, syscall_clock_gettime64<W>);
	// clock_getres
	install_syscall_handler(114, syscall_stub_nosys<W>);
	// clock_nanosleep
	install_syscall_handler(115, syscall_clock_nanosleep<W>);
	// sched_getaffinity
	install_syscall_handler(123, syscall_stub_nosys<W>);
	// tkill
	install_syscall_handler(130,
	[] (Machine<W>& machine) {
		const int tid = machine.template sysarg<int> (0);
		const int sig = machine.template sysarg<int> (1);
		SYSPRINT(">>> tkill on tid=%d signal=%d\n", tid, sig);
		(void) tid;
		// If the signal zero or unset, ignore it
		if (sig == 0 || machine.sigaction(sig).is_unset()) {
			return;
		} else {
			// Jump to signal handler and change to altstack, if set
			machine.signals().enter(machine, sig);
			SYSPRINT("<<< tkill signal=%d jumping to 0x%lX (sp=0x%lX)\n",
				sig, (long)machine.cpu.pc(), (long)machine.cpu.reg(REG_SP));
			return;
		}
		machine.stop();
	});
	// sigaltstack
	install_syscall_handler(132, syscall_sigaltstack<W>);
	// rt_sigaction
	install_syscall_handler(134, syscall_sigaction<W>);
	// rt_sigprocmask
	install_syscall_handler(135, syscall_stub_zero<W>);
	// uname
	install_syscall_handler(160, syscall_uname<W>);
	// prctl
	install_syscall_handler(167, syscall_stub_nosys<W>);
	// gettimeofday
	install_syscall_handler(169, syscall_gettimeofday<W>);
	// getpid
	install_syscall_handler(172, syscall_stub_zero<W>);
	// getuid
	install_syscall_handler(174, syscall_stub_zero<W>);
	// geteuid
	install_syscall_handler(175, syscall_stub_zero<W>);
	// getgid
	install_syscall_handler(176, syscall_stub_zero<W>);
	// getegid
	install_syscall_handler(177, syscall_stub_zero<W>);

	install_syscall_handler(214, syscall_brk<W>);

	// msync
	install_syscall_handler(227, syscall_stub_zero<W>);

	// riscv_hwprobe
	install_syscall_handler(258, syscall_stub_zero<W>);
	// riscv_flush_icache
	install_syscall_handler(259, syscall_stub_zero<W>);

	install_syscall_handler(278, syscall_getrandom<W>);

#if defined(__linux__) && !defined(__ANDROID__)
	// statx
	install_syscall_handler(291, syscall_statx<W>);
#endif
	// rseq
	install_syscall_handler(293, syscall_stub_nosys<W>);

	add_mman_syscalls<W>();

	if (filesystem || sockets) {
		// Workaround for a broken "feature"
		// Closing sockets that are already closed cause SIGPIPE signal
		signal(SIGPIPE, SIG_IGN);

		m_fds.reset(new FileDescriptors);
		if (sockets)
			add_socket_syscalls(*this);
	}

	register_clobbering_syscall(130); // TKILL
	register_clobbering_syscall(93);  // EXIT
	register_clobbering_syscall(94);  // EXIT_GROUP
}

#ifdef RISCV_32I
template void Machine<4>::setup_newlib_syscalls();
template void Machine<4>::setup_newlib_syscalls(bool filesystem);
template void Machine<4>::setup_linux_syscalls(bool, bool);
#endif
#ifdef RISCV_64I
template void Machine<8>::setup_newlib_syscalls();
template void Machine<8>::setup_newlib_syscalls(bool filesystem);
template void Machine<8>::setup_linux_syscalls(bool, bool);
#endif

FileDescriptors::~FileDescriptors() {
	// Close all the real FDs
	for (const auto& it : translation) {
		::close(it.second);
	}
}

} // riscv
