#include <libriscv/machine.hpp>

#include <libriscv/threads.hpp>
#include <chrono>
#include <cstdint>
#include <io.h>

typedef std::make_signed_t<size_t> ssize_t;
#ifndef PATH_MAX
static constexpr size_t PATH_MAX = 512;
#endif

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

#include <winsock2.h>
#include <sys/stat.h>

#define SA_ONSTACK	0x08000000

namespace riscv {
template<int W>
void add_socket_syscalls(Machine<W> &);

template<int W>
struct guest_iovec {
	address_type<W> iov_base;
	address_type<W> iov_len;
};

template<int W>
static void syscall_stub_zero(Machine<W> &machine) {
	SYSPRINT("SYSCALL stubbed (zero): %d\n", (int) machine.cpu.reg(17));
	machine.set_result(0);
}

template<int W>
static void syscall_stub_nosys(Machine<W> &machine) {
	SYSPRINT("SYSCALL stubbed (nosys): %d\n", (int) machine.cpu.reg(17));
	machine.set_result(-ENOSYS);
}

template<int W>
static void syscall_exit(Machine<W> &machine) {
	// Stop sets the max instruction counter to zero, making the instruction loop end.
	machine.stop();
}

template<int W>
static void syscall_ebreak(riscv::Machine<W> &machine) {
	printf("\n>>> EBREAK at %#lX\n", (long) machine.cpu.pc());
#ifdef RISCV_DEBUG
	machine.print_and_pause();
#else
	throw MachineException(UNHANDLED_SYSCALL, "EBREAK instruction");
#endif
}

template<int W>
static void syscall_sigaltstack(Machine<W> &machine) {
	const auto ss = machine.sysarg(0);
	const auto old_ss = machine.sysarg(1);
	SYSPRINT("SYSCALL sigaltstack, tid=%d ss: 0x%lX old_ss: 0x%lX\n",
			 machine.gettid(), (long) ss, (long) old_ss);

	auto &stack = machine.signals().per_thread(machine.gettid()).stack;

	if (old_ss != 0x0) {
		machine.copy_to_guest(old_ss, &stack, sizeof(stack));
	}
	if (ss != 0x0) {
		machine.copy_from_guest(&stack, ss, sizeof(stack));

		SYSPRINT("<<< sigaltstack sp: 0x%lX flags: 0x%X size: 0x%lX\n",
				 (long) stack.ss_sp, stack.ss_flags, (long) stack.ss_size);
	}

	machine.set_result(0);
}

template<int W>
static void syscall_sigaction(Machine<W> &machine) {
	const int sig = machine.sysarg(0);
	const auto action = machine.sysarg(1);
	const auto old_action = machine.sysarg(2);
	SYSPRINT("SYSCALL sigaction, signal: %d, action: 0x%lX old_action: 0x%lX\n",
			 sig, (long) action, (long) old_action);
	if (sig == 0) return;

	auto& sigact = machine.sigaction(sig);

	struct kernel_sigaction {
		address_type<W> sa_handler;
		address_type<W> sa_flags;
		address_type<W> sa_mask;
	} sa{};
	if (old_action != 0x0) {
		sa.sa_handler = sigact.handler & ~address_type<W>(0xF);
		sa.sa_flags = (sigact.altstack ? SA_ONSTACK : 0x0);
		sa.sa_mask = sigact.mask;
		machine.copy_to_guest(old_action, &sa, sizeof(sa));
	}
	if (action != 0x0) {
		machine.copy_from_guest(&sa, action, sizeof(sa));
		sigact.handler = sa.sa_handler;
		sigact.altstack = (sa.sa_flags & SA_ONSTACK) != 0;
		sigact.mask = sa.sa_mask;
		SYSPRINT("<<< sigaction %d handler: 0x%lX altstack: %d\n",
			sig, (long)sigact.handler, sigact.altstack);
	}

	machine.set_result(0);
}

template<int W>
void syscall_lseek(Machine<W> &machine) {
	const int fd = machine.template sysarg<int>(0);
	const auto offset = machine.template sysarg<int64_t>(1);
	const int whence = machine.template sysarg<int>(2);
	SYSPRINT("SYSCALL lseek, fd: %d, offset: 0x%lX, whence: %d\n",
			 fd, (long) offset, whence);

	const auto real_fd = machine.fds().get(fd);
	if (machine.fds().is_socket(fd)) {
		machine.set_result(-ESPIPE);
	} else {
		int64_t res = _lseek(real_fd, offset, whence);
		if (res >= 0) {
			machine.set_result(res);
		} else {
			machine.set_result(-errno);
		}
	}
}

template<int W>
static void syscall_read(Machine<W> &machine) {
	const int fd = machine.template sysarg<int>(0);
	const auto address = machine.sysarg(1);
	const size_t len = machine.sysarg(2);
	SYSPRINT("SYSCALL read, addr: 0x%lX, len: %zu\n", (long) address, len);
	// We have special stdin handling
	if (fd == 0) {
		// Arbitrary maximum read length
		if (len > 1024 * 1024 * 16) {
			machine.set_result(-ENOMEM);
			return;
		}
		auto buffer = std::unique_ptr<char[]>(new char[len]);
		long result = machine.stdin_read(buffer.get(), len);
		if (result > 0) {
			machine.copy_to_guest(address, buffer.get(), result);
		}
		machine.set_result(result);
		return;
	} else if (machine.has_file_descriptors()) {
		const auto real_fd = machine.fds().get(fd);
		// Gather up to 1MB of pages we can read into
		riscv::vBuffer buffers[256];
		size_t cnt =
				machine.memory.gather_buffers_from_range(256, buffers, address, len);

		size_t bytes = 0;
		for (size_t i = 0; i < cnt; i++) {
			ssize_t res;
			if (machine.fds().is_socket(fd))
				res = recv(real_fd, buffers[i].ptr, buffers[i].len, 0);
			else
				res = _read(real_fd, buffers[i].ptr, buffers[i].len);
			if (res >= 0) {
				bytes += res;
				if ((size_t) res < buffers[i].len) break;
			} else {
				machine.set_result_or_error(res);
				return;
			}
		}
		machine.set_result(bytes);
		return;
	}
	machine.set_result(-EBADF);
}

template<int W>
static void syscall_write(Machine<W> &machine) {
	const int vfd = machine.template sysarg<int>(0);
	const auto address = machine.sysarg(1);
	const size_t len = machine.sysarg(2);
	SYSPRINT("SYSCALL write, fd: %d addr: 0x%lX, len: %zu\n",
			 vfd, (long) address, len);
	// We only accept standard output pipes, for now :)
	if (vfd == 1 || vfd == 2) {
		// Zero-copy retrieval of buffers (16 fragments)
		riscv::vBuffer buffers[16];
		size_t cnt =
				machine.memory.gather_buffers_from_range(16, buffers, address, len);
		for (size_t i = 0; i < cnt; i++) {
			machine.print(buffers[i].ptr, buffers[i].len);
		}
		machine.set_result(len);
		return;
	} else if (machine.has_file_descriptors() && machine.fds().permit_write(vfd)) {
		if (vfd >= 123456780) {
			machine.set_result(len);
			return;
		}

		auto real_fd = machine.fds().get(vfd);
		// Zero-copy retrieval of buffers (64 fragments)
		riscv::vBuffer buffers[64];
		size_t cnt =
				machine.memory.gather_buffers_from_range(64, buffers, address, len);
		size_t bytes = 0;
		// Could probably be a writev call, tbh
		for (size_t i = 0; i < cnt; i++) {
			ssize_t res;
			if (machine.fds().is_socket(vfd))
				res = send(real_fd, buffers[i].ptr, buffers[i].len, 0);
			else
				res = _write(real_fd, buffers[i].ptr, buffers[i].len);
			if (res >= 0) {
				bytes += res;
				// Detect partial writes
				if ((size_t) res < buffers[i].len) break;
			} else {
				// Detect write errors
				machine.set_result_or_error(res);
				return;
			}
		}
		machine.set_result(bytes);
		return;
	}
	machine.set_result(-EBADF);
}

template<int W>
static void syscall_writev(Machine<W> &machine) {
	const int fd = machine.template sysarg<int>(0);
	const auto iov_g = machine.sysarg(1);
	const auto count = machine.template sysarg<int>(2);
	if constexpr (false) {
		printf("SYSCALL writev, iov: %#X  cnt: %d\n", iov_g, count);
	}
	if (count < 0 || count > 256) {
		machine.set_result(-EINVAL);
		return;
	}
	// We only accept standard output pipes, for now :)
	if (fd == 1 || fd == 2) {
		const size_t size = sizeof(guest_iovec<W>) * count;

		std::vector<guest_iovec<W>> vec(count);
		machine.memory.memcpy_out(vec.data(), iov_g, size);

		ssize_t res = 0;
		for (const auto &iov: vec) {
			auto src_g = (address_type<W>) iov.iov_base;
			auto len_g = (size_t) iov.iov_len;
			/* Zero-copy retrieval of buffers */
			riscv::vBuffer buffers[4];
			size_t cnt =
					machine.memory.gather_buffers_from_range(4, buffers, src_g, len_g);
			for (size_t i = 0; i < cnt; i++) {
				machine.print(buffers[i].ptr, buffers[i].len);
			}
			res += len_g;
		}
		machine.set_result(res);
		return;
	}
	machine.set_result(-EBADF);
}

template<int W>
static void syscall_openat(Machine<W> &machine) {
	const int dir_fd = machine.template sysarg<int>(0);
	const auto g_path = machine.sysarg(1);
	const int flags = machine.template sysarg<int>(2);

	std::string path = machine.memory.memstring(g_path);

	SYSPRINT("SYSCALL openat, dir_fd: %d path: %s flags: %X\n",
			 dir_fd, path.c_str(), flags);

	if (machine.has_file_descriptors() && machine.fds().permit_filesystem) {

		if (machine.fds().filter_open != nullptr) {
			if (!machine.fds().filter_open(machine.template get_userdata<void>(), path)) {
				machine.set_result(-EPERM);
				return;
			}
		}
		/*
		int real_fd = openat(machine.fds().translate(dir_fd), path, flags);
		if (real_fd > 0) {
			const int vfd = machine.fds().assign_file(real_fd);
			machine.set_result(vfd);
		} else {
			// Translate errno() into kernel API return value
			machine.set_result(-errno);
		}*/
		(void)dir_fd;
		(void)flags;
		machine.set_result(-EPERM);
		return;
	}

	machine.set_result(-EBADF);
}

template<int W>
static void syscall_close(riscv::Machine<W> &machine) {
	const int vfd = machine.template sysarg<int>(0);
	if constexpr (verbose_syscalls) {
		printf("SYSCALL close, fd: %d\n", vfd);
	}

	if (vfd >= 0 && vfd <= 2) {
		// TODO: Do we really want to close them?
		machine.set_result(0);
		return;
	} else if (machine.has_file_descriptors()) {
		const auto res = machine.fds().erase(vfd);
		if (res > 0) {
			if (machine.fds().is_socket(vfd)) {
				closesocket(res);
			} else {
				_close(res);
			}
		}
		machine.set_result(res >= 0 ? 0 : -EBADF);
		return;
	}
	machine.set_result(-EBADF);
}

template<int W>
static void syscall_dup(Machine<W> &machine) {
	const int vfd = machine.template sysarg<int>(0);
	SYSPRINT("SYSCALL dup, fd: %d\n", vfd);
	(void)vfd;

	machine.set_result(-EBADF);
}

template<int W>
static void syscall_fcntl(Machine<W> &machine) {
	const int vfd = machine.template sysarg<int>(0);
	const auto cmd = machine.template sysarg<int>(1);
	const auto arg1 = machine.sysarg(2);
	const auto arg2 = machine.sysarg(3);
	const auto arg3 = machine.sysarg(4);
	SYSPRINT("SYSCALL fcntl, fd: %d  cmd: 0x%X\n", vfd, cmd);

	if (machine.has_file_descriptors()) {

		// Emulate fcntl for stdin/stdout/stderr
		if (vfd == 0 || vfd == 1 || vfd == 2) {
			machine.set_result(0);
			return;
		}

		/*int real_fd = machine.fds().translate(vfd);
		int res = fcntl(real_fd, cmd, arg1, arg2, arg3);
		machine.set_result_or_error(res);*/
		(void)vfd;
		(void)cmd;
		(void)arg1;
		(void)arg2;
		(void)arg3;
		machine.set_result(-EPERM);
		return;
	}
	machine.set_result(-EBADF);
}

template<int W>
static void syscall_ioctl(Machine<W> &machine) {
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

		/*int real_fd = machine.fds().translate(vfd);
		int res = ioctl(real_fd, req, arg1, arg2, arg3, arg4);
		machine.set_result_or_error(res);*/
		(void)vfd;
		(void)req;
		(void)arg1;
		(void)arg2;
		(void)arg3;
		(void)arg4;
		machine.set_result(-EPERM);
		return;
	}
	machine.set_result(-EBADF);
}

template<int W>
void syscall_readlinkat(Machine<W> &machine) {
	const int vfd = machine.template sysarg<int>(0);
	const auto g_path = machine.sysarg(1);
	const auto g_buf = machine.sysarg(2);
	const auto bufsize = machine.sysarg(3);

	const std::string original_path = machine.memory.memstring(g_path);

	SYSPRINT("SYSCALL readlinkat, fd: %d path: %s buffer: 0x%lX size: %zu\n",
			 vfd, original_path.c_str(), (long) g_buf, (size_t) bufsize);

	char buffer[16384];
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
		const auto real_fd = machine.fds().translate(vfd);

		/*const int res = readlinkat(real_fd, original_path.c_str(), buffer, bufsize);
		if (res > 0) {
			// TODO: Only necessary if g_buf is not sequential.
			machine.copy_to_guest(g_buf, buffer, res);
		}

		machine.set_result_or_error(res);*/
		(void)real_fd;
		(void)g_buf;
		(void)bufsize;
		machine.set_result(-ENOSYS);
		return;
	}
	machine.set_result(-ENOSYS);
}

// The RISC-V stat structure is different from x86
struct riscv_stat {
	uint64_t st_dev;        /* Device.  */
	uint64_t st_ino;        /* File serial number.  */
	uint32_t st_mode;    /* File mode.  */
	uint32_t st_nlink;    /* Link count.  */
	uint32_t st_uid;        /* User ID of the file's owner.  */
	uint32_t st_gid;        /* Group ID of the file's group. */
	uint64_t st_rdev;    /* Device number, if device.  */
	uint64_t __pad1;
	int64_t st_size;    /* Size of file, in bytes.  */
	int32_t st_blksize;    /* Optimal block size for I/O.  */
	int32_t __pad2;
	int64_t st_blocks;    /* Number 512-byte blocks allocated. */
	int64_t rv_atime;    /* Time of last access.  */
	uint64_t rv_atime_nsec;
	int64_t rv_mtime;    /* Time of last modification.  */
	uint64_t rv_mtime_nsec;
	int64_t rv_ctime;    /* Time of last status change.  */
	uint64_t rv_ctime_nsec;
	uint32_t __unused4;
	uint32_t __unused5;
};

inline void copy_stat_buffer(struct stat &st, struct riscv_stat &rst) {
	rst.st_dev = st.st_dev;
	rst.st_ino = st.st_ino;
	rst.st_mode = st.st_mode;
	rst.st_nlink = st.st_nlink;
	rst.st_uid = st.st_uid;
	rst.st_gid = st.st_gid;
	rst.st_rdev = st.st_rdev;
	rst.st_size = st.st_size;
	rst.st_blksize = 512;
	rst.st_blocks = (st.st_size - 1) / 512 + 1;
	rst.rv_atime = st.st_atime;
	rst.rv_atime_nsec = 0;
	rst.rv_mtime = st.st_mtime;
	rst.rv_mtime_nsec = 0;
	rst.rv_ctime = st.st_ctime;
	rst.rv_ctime_nsec = 0;
}

template<int W>
static void syscall_fstatat(Machine<W> &machine) {
	const auto vfd = machine.template sysarg<int>(0);
	const auto g_path = machine.sysarg(1);
	const auto g_buf = machine.sysarg(2);
	const auto flags = machine.template sysarg<int>(3);

	char path[PATH_MAX];
	machine.copy_from_guest(path, g_path, sizeof(path) - 1);
	path[sizeof(path) - 1] = 0;

	SYSPRINT("SYSCALL fstatat, fd: %d path: %s buf: 0x%lX flags: %#x)\n",
			 vfd, path, (long) g_buf, flags);

	if (machine.has_file_descriptors()) {

		if (vfd == 0 || vfd == 1 || vfd == 2) {
			// Emulate stat for stdin/stdout/stderr
			struct riscv_stat rst;
			std::memset(&rst, 0, sizeof(rst));
			rst.st_mode = 0x2000; // S_IFCHR
			rst.st_dev = 0;
			rst.st_ino = 0;
			rst.st_nlink = 1;
			rst.st_uid = 0;
			rst.st_gid = 0;
			rst.st_rdev = 0;
			rst.st_size = 0;
			rst.st_blksize = 0;
			rst.st_blocks = 0;
			rst.rv_atime = 0;
			rst.rv_mtime = 0;
			rst.rv_ctime = 0;
			machine.copy_to_guest(g_buf, &rst, sizeof(rst));
			machine.set_result(0);
			return;
		}

		auto real_fd = machine.fds().translate(vfd);

		/*struct stat st;
		const int res = ::fstatat(real_fd, path, &st, flags);
		if (res == 0) {
			// Convert to RISC-V structure
			struct riscv_stat rst;
			copy_stat_buffer(st, rst);
			machine.copy_to_guest(g_buf, &rst, sizeof(rst));
		}
		machine.set_result_or_error(res);*/
		(void)real_fd;
		(void)g_buf;
		(void)flags;
		machine.set_result(-ENOSYS);
		return;
	}
	machine.set_result(-ENOSYS);
}

template<int W>
static void syscall_fstat(Machine<W> &machine) {
	const auto vfd = machine.template sysarg<int>(0);
	const auto g_buf = machine.sysarg(1);

	SYSPRINT("SYSCALL fstat, fd: %d buf: 0x%lX)\n",
			 vfd, (long) g_buf);

	if (machine.has_file_descriptors()) {

		auto real_fd = machine.fds().translate(vfd);

		struct stat st;
		int res = ::fstat(real_fd, &st);
		if (res == 0) {
			// Convert to RISC-V structure
			struct riscv_stat rst;
			copy_stat_buffer(st, rst);
			machine.copy_to_guest(g_buf, &rst, sizeof(rst));
		}
		machine.set_result_or_error(res);
		return;
	}
	machine.set_result(-ENOSYS);
}

template<int W>
static void syscall_statx(Machine<W> &machine) {
	const int dir_fd = machine.template sysarg<int>(0);
	const auto g_path = machine.sysarg(1);
	const int flags = machine.template sysarg<int>(2);
	const auto mask = machine.template sysarg<uint32_t>(3);
	const auto buffer = machine.sysarg(4);

	char path[PATH_MAX];
	machine.copy_from_guest(path, g_path, sizeof(path) - 1);
	path[sizeof(path) - 1] = 0;

	SYSPRINT("SYSCALL statx, fd: %d path: %s flags: %x buf: 0x%lX)\n",
			 dir_fd, path, flags, (long) buffer);

	if (machine.has_file_descriptors()) {
		if (machine.fds().filter_stat != nullptr) {
			if (!machine.fds().filter_stat(machine.template get_userdata<void>(), path)) {
				machine.set_result(-EPERM);
				return;
			}
		}

		/*struct statx st;
		int res = ::statx(dir_fd, path, flags, mask, &st);
		if (res == 0) {
			machine.copy_to_guest(buffer, &st, sizeof(struct statx));
		}
		machine.set_result_or_error(res);*/
		(void)dir_fd;
		(void)flags;
		(void)mask;
		(void)buffer;
		machine.set_result(-ENOSYS);
		return;
	}
	machine.set_result(-ENOSYS);
}

template<int W>
static void syscall_gettimeofday(Machine<W> &machine) {
	const auto buffer = machine.sysarg(0);
	SYSPRINT("SYSCALL gettimeofday, buffer: 0x%lX\n", (long) buffer);
	auto tp = std::chrono::system_clock::now();
	auto secs = std::chrono::time_point_cast<std::chrono::seconds>(tp);
	auto us = std::chrono::time_point_cast<std::chrono::microseconds>(tp) -
		time_point_cast<std::chrono::microseconds>(secs);
	struct timeval tv =
		timeval{ long(secs.time_since_epoch().count()), long(us.count()) };
	if constexpr (W == 4) {
		int32_t timeval32[2] = {(int) tv.tv_sec, (int) tv.tv_usec};
		machine.copy_to_guest(buffer, timeval32, sizeof(timeval32));
	} else {
		machine.copy_to_guest(buffer, &tv, sizeof(tv));
	}
	machine.set_result_or_error(0);
}

template<int W>
static void syscall_clock_gettime(Machine<W>& machine) {
	const auto clkid = machine.template sysarg<int>(0);
	const auto buffer = machine.sysarg(1);
	SYSPRINT("SYSCALL clock_gettime, clkid: %x buffer: 0x%lX\n",
			 clkid, (long) buffer);

	auto tp = std::chrono::system_clock::now();
	auto secs = std::chrono::time_point_cast<std::chrono::seconds>(tp);
	auto ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(tp) -
		time_point_cast<std::chrono::nanoseconds>(secs);
	struct timespec ts =
		timespec{ secs.time_since_epoch().count(), long(ns.count()) };
	(void)clkid;

	if constexpr (W == 4) {
		int32_t ts32[2] = {(int) ts.tv_sec, (int) ts.tv_nsec};
		machine.copy_to_guest(buffer, &ts32, sizeof(ts32));
	} else {
		machine.copy_to_guest(buffer, &ts, sizeof(ts));
	}

	machine.set_result_or_error(0);
}
template<int W>
static void syscall_clock_gettime64(Machine<W>& machine) {
	const auto clkid = machine.template sysarg<int>(0);
	const auto buffer = machine.sysarg(1);
	SYSPRINT("SYSCALL clock_gettime64, clkid: %x buffer: 0x%lX\n",
			 clkid, (long) buffer);

	auto tp = std::chrono::system_clock::now();
	auto secs = std::chrono::time_point_cast<std::chrono::seconds>(tp);
	auto ms = std::chrono::time_point_cast<std::chrono::milliseconds>(tp) -
		time_point_cast<std::chrono::milliseconds>(secs);
	struct timespec ts =
		timespec{ secs.time_since_epoch().count(), long(ms.count()) };
	(void)clkid;

	machine.copy_to_guest(buffer, &ts, sizeof(ts));
	machine.set_result_or_error(0);
}

template<int W>
static void syscall_uname(Machine<W> &machine) {
	const auto buffer = machine.sysarg(0);
	SYSPRINT("SYSCALL uname, buffer: 0x%lX\n", (long) buffer);
	static constexpr int UTSLEN = 65;
	struct {
		char sysname[UTSLEN];
		char nodename[UTSLEN];
		char release[UTSLEN];
		char version[UTSLEN];
		char machine[UTSLEN];
		char domain[UTSLEN];
	} uts;
	strcpy_s(uts.sysname, UTSLEN, "RISC-V C++ Emulator");
	strcpy_s(uts.nodename, UTSLEN, "libriscv");
	strcpy_s(uts.release, UTSLEN, "5.6.0");
	strcpy_s(uts.version, UTSLEN, "");
	if constexpr (W == 4)
		strcpy_s(uts.machine, UTSLEN, "rv32imafdc");
	else if constexpr (W == 8)
		strcpy_s(uts.machine, UTSLEN, "rv64imafdc");
	else
		strcpy_s(uts.machine, UTSLEN, "rv128imafd");
	strcpy_s(uts.domain, UTSLEN, "(none)");

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

template<int W>
static void syscall_brk(Machine<W> &machine) {
	auto new_end = machine.sysarg(0);
	if (new_end > machine.memory.heap_address() + Memory<W>::BRK_MAX) {
		new_end = machine.memory.heap_address() + Memory<W>::BRK_MAX;
	} else if (new_end < machine.memory.heap_address()) {
		new_end = machine.memory.heap_address();
	}

	if constexpr (verbose_syscalls) {
		printf("SYSCALL brk, new_end: 0x%lX\n", (long) new_end);
	}
	machine.set_result(new_end);
}

template <int W>
static void syscall_pipe2(Machine<W>& machine)
{
	const auto vfd_array = machine.sysarg(0);

	if (machine.has_file_descriptors()) {
		int vpipes[2];
		vpipes[0] = 123456784;
		vpipes[1] = 123456785;
		machine.copy_to_guest(vfd_array, vpipes, sizeof(vpipes));
		machine.set_result(0);
	} else {
		machine.set_result(-1);
	}
	SYSPRINT("SYSCALL pipe2, fd array: 0x%lX flags: %d = %ld\n",
		(long)vfd_array, flags, (long)machine.return_value());
}

template <int W>
static void syscall_getrandom(Machine<W>& machine)
{
	const auto g_addr = machine.sysarg(0);
	const auto g_len  = machine.sysarg(1);

	std::array<uint8_t, 256> buffer;
	if (g_len > buffer.size()) {
		machine.set_result(-1);
		return;
	}
	const size_t need = std::min((size_t)g_len, buffer.size());
	for (size_t i = 0; i < need; ++i) {
		buffer[i] ^= rand() & 0xFF; // XXX: Not secure
	}
	const ssize_t result = need;
	if (result > 0) {
		machine.copy_to_guest(g_addr, buffer.data(), result);
		// getrandom() is a slow syscall, penalize it
		machine.penalize(20'000 * result); // 20K insn per byte
	}
	machine.set_result(result);

	if constexpr (verbose_syscalls) {
		printf("SYSCALL getrandom(addr=0x%lX, len=%ld) = %ld\n",
			(long)g_addr, (long)g_len, (long)machine.return_value());
	}
}

#include "../linux/syscalls_mman.cpp"
#include "epoll.cpp"

template<int W>
void Machine<W>::setup_minimal_syscalls() {
	install_syscall_handler(SYSCALL_EBREAK, syscall_ebreak<W>);
	install_syscall_handler(62, syscall_lseek<W>);
	install_syscall_handler(63, syscall_read<W>);
	install_syscall_handler(64, syscall_write<W>);
	install_syscall_handler(93, syscall_exit<W>);
}

template<int W>
void Machine<W>::setup_newlib_syscalls() {
	setup_minimal_syscalls();
	install_syscall_handler(169, syscall_gettimeofday<W>);
	install_syscall_handler(214, syscall_brk<W>);
	add_mman_syscalls<W>();
}
template<int W>
void Machine<W>::setup_newlib_syscalls(bool) {
	setup_newlib_syscalls();
}

template<int W>
void Machine<W>::setup_linux_syscalls(bool filesystem, bool sockets) {
	this->setup_minimal_syscalls();

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
	// fcntl
	install_syscall_handler(25, syscall_fcntl<W>);
	// ioctl
	install_syscall_handler(29, syscall_ioctl<W>);
	// faccessat
	install_syscall_handler(48, syscall_stub_nosys<W>);

	install_syscall_handler(56, syscall_openat<W>);
	install_syscall_handler(57, syscall_close<W>);
	install_syscall_handler(59, syscall_pipe2<W>);
	install_syscall_handler(66, syscall_writev<W>);
	// 73: ppoll
	install_syscall_handler(73, syscall_stub_zero<W>);
	install_syscall_handler(78, syscall_readlinkat<W>);
	// 79: fstatat
	install_syscall_handler(79, syscall_fstatat<W>);
	// 80: fstat
	install_syscall_handler(80, syscall_fstat<W>);
	// 90: capget
	install_syscall_handler(90, syscall_capget<W>);

	// 94: exit_group (single-threaded)
	install_syscall_handler(94, syscall_exit<W>);

	// nanosleep
	install_syscall_handler(101, syscall_stub_zero<W>);
	// clock_gettime
	install_syscall_handler(113, syscall_clock_gettime<W>);
	install_syscall_handler(403, syscall_clock_gettime64<W>);
	// clock_getres
	install_syscall_handler(114, syscall_stub_nosys<W>);
	// sched_getaffinity
	install_syscall_handler(123, syscall_stub_nosys<W>);
	// sigaltstack
	install_syscall_handler(132, syscall_sigaltstack<W>);
	// rt_sigaction
	install_syscall_handler(134, syscall_sigaction<W>);
	// rt_sigprocmask
	install_syscall_handler(135, syscall_stub_zero<W>);

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
	// getegid
	install_syscall_handler(177, syscall_stub_zero<W>);

	install_syscall_handler(160, syscall_uname<W>);
	install_syscall_handler(214, syscall_brk<W>);
	// riscv_hwprobe
	install_syscall_handler(258, syscall_stub_zero<W>);
	// riscv_flush_icache
	install_syscall_handler(259, syscall_stub_zero<W>);

	install_syscall_handler(278, syscall_getrandom<W>);

	// statx
	install_syscall_handler(291, syscall_statx<W>);
	// rseq
	install_syscall_handler(293, syscall_stub_nosys<W>);

	add_mman_syscalls<W>();

	if (filesystem || sockets) {
		m_fds.reset(new FileDescriptors);
		if (sockets)
			add_socket_syscalls(*this);
	}
}

#ifdef RISCV_32I
template void Machine<4>::setup_minimal_syscalls();
template void Machine<4>::setup_newlib_syscalls();
template void Machine<4>::setup_newlib_syscalls(bool);
template void Machine<4>::setup_linux_syscalls(bool, bool);
#endif
#ifdef RISCV_64I
template void Machine<8>::setup_minimal_syscalls();
template void Machine<8>::setup_newlib_syscalls();
template void Machine<8>::setup_newlib_syscalls(bool);
template void Machine<8>::setup_linux_syscalls(bool, bool);
#endif

FileDescriptors::~FileDescriptors() {
	// Close all the real FDs
	for (const auto &it: translation) {
		if (is_socket(it.first)) {
			closesocket(it.second);
		} else {
			_close(it.second);
		}
	}
}

} // riscv
