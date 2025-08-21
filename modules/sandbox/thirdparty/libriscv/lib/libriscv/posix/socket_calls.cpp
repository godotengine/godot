#include <libriscv/machine.hpp>

//#define SOCKETCALL_VERBOSE 1
#ifdef SOCKETCALL_VERBOSE
#define SYSPRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define SYSPRINT(fmt, ...) /* fmt */
#endif

#ifndef _WIN32
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#if __has_include(<linux/netlink.h>)
#define HAVE_LINUX_NETLINK
#include <linux/netlink.h>
#endif
#else
#include "../win32/ws2.hpp"
WSADATA riscv::ws2::global_winsock_data;
bool riscv::ws2::winsock_initialized = false;
using ssize_t = long long int;
#undef min
#undef max
#endif

namespace riscv {

template <int W>
struct guest_iovec {
	address_type<W> iov_base;
	address_type<W> iov_len;
};
template <int W>
struct guest_msghdr
{
	address_type<W> msg_name;		/* Address to send to/receive from.  */
	uint32_t        msg_namelen;	/* Length of address data.  */

	address_type<W> msg_iov;		/* Vector of data to send/receive into.  */
	address_type<W> msg_iovlen;		/* Number of elements in the vector.  */

	address_type<W> msg_control;	/* Ancillary data (eg BSD filedesc passing). */
	address_type<W> msg_controllen;	/* Ancillary data buffer length. */
	int             msg_flags;		/* Flags on received message.  */
};

#ifdef SOCKETCALL_VERBOSE
template <int W>
static void print_address(std::array<char, 128> buffer, address_type<W> addrlen)
{
	if (addrlen < 12)
		throw MachineException(INVALID_PROGRAM, "Socket address too small", addrlen);

	char printbuf[INET6_ADDRSTRLEN];
	auto* sin6 = (struct sockaddr_in6 *)buffer.data();

	switch (sin6->sin6_family) {
	case AF_INET6:
		inet_ntop(AF_INET6, &sin6->sin6_addr, printbuf, sizeof(printbuf));
		printf("SYSCALL -- IPv6 address: %s\n", printbuf);
		break;
	case AF_INET: {
		auto* sin4 = (struct sockaddr_in *)buffer.data();
		inet_ntop(AF_INET, &sin4->sin_addr, printbuf, sizeof(printbuf));
		printf("SYSCALL -- IPv4 address: %s\n", printbuf);
		break;
	}
	case AF_UNIX: {
		auto* sun = (struct sockaddr_un *)buffer.data();
		printf("SYSCALL -- UNIX address: %s\n", sun->sun_path);
		break;
	}
#ifdef HAVE_LINUX_NETLINK
	case AF_NETLINK: {
		auto* snl = (struct sockaddr_nl *)buffer.data();
		printf("SYSCALL -- NetLink port: %u\n", snl->nl_pid);
		break;
	}
#endif
	}
}
#endif

template <int W>
static void syscall_socket(Machine<W>& machine)
{
	const auto [domain, type, proto] =
		machine.template sysargs<int, int, int> ();
	int real_fd = -1;

	if (machine.has_file_descriptors() && machine.fds().permit_sockets) {
#ifdef WIN32
        ws2::init();
#endif
		real_fd = socket(domain, type, proto);
		if (real_fd > 0) {
			const int vfd = machine.fds().assign_socket(real_fd);
			machine.set_result(vfd);
		} else {
			// Translate errno() into kernel API return value
			machine.set_result(-errno);
		}
	} else {
		machine.set_result(-EBADF);
	}
#ifdef SOCKETCALL_VERBOSE
	const char* domname;
	switch (domain & 0xFF) {
		case AF_UNIX: domname = "Unix"; break;
		case AF_INET: domname = "IPv4"; break;
		case AF_INET6: domname = "IPv6"; break;
#ifdef HAVE_LINUX_NETLINK
		case AF_NETLINK: domname = "Netlink"; break;
#endif
		default: domname = "unknown";
	}
	const char* typname;
	switch (type & 0xFF) {
		case SOCK_STREAM: typname = "Stream"; break;
		case SOCK_DGRAM: typname = "Datagram"; break;
		case SOCK_SEQPACKET: typname = "Seq.packet"; break;
		case SOCK_RAW: typname = "Raw"; break;
		default: typname = "unknown";
	}
	SYSPRINT("SYSCALL socket, domain: %x (%s) type: %x (%s) proto: %x = %d (real fd: %d)\n",
		domain, domname, type, typname, proto, (int)machine.return_value(), real_fd);
#endif
}

template <int W>
static void syscall_bind(Machine<W>& machine)
{
	const auto [vfd, g_addr, addrlen] =
		machine.template sysargs<int, address_type<W>, address_type<W>> ();

	alignas(16) std::array<char, 128> buffer;
	if (addrlen > buffer.size()) {
		machine.set_result(-ENOMEM);
		return;
	}

	if (machine.has_file_descriptors() && machine.fds().permit_sockets) {

		const auto real_fd = machine.fds().translate(vfd);
		machine.copy_from_guest(buffer.data(), g_addr, addrlen);

	#ifdef SOCKETCALL_VERBOSE
		print_address<W>(buffer, addrlen);
	#endif

		int res = bind(real_fd, (struct sockaddr *)buffer.data(), addrlen);
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}

	SYSPRINT("SYSCALL bind, vfd: %d addr: 0x%lX len: 0x%lX = %d\n",
		vfd, (long)g_addr, (long)addrlen, (int)machine.return_value());
}

template <int W>
static void syscall_listen(Machine<W>& machine)
{
	const auto [vfd, backlog] =
		machine.template sysargs<int, int> ();

	SYSPRINT("SYSCALL listen, vfd: %d backlog: %d\n",
		vfd, backlog);

	if (machine.has_file_descriptors() && machine.fds().permit_sockets) {

		const auto real_fd = machine.fds().translate(vfd);

		int res = listen(real_fd, backlog);
		machine.set_result_or_error(res);
		return;
	}
	machine.set_result(-EBADF);
}

template <int W>
static void syscall_accept(Machine<W>& machine)
{
	const auto [vfd, g_addr, g_addrlen] =
		machine.template sysargs<int, address_type<W>, address_type<W>> ();

	SYSPRINT("SYSCALL accept, vfd: %d addr: 0x%lX\n",
		vfd, (long)g_addr);

	if (machine.has_file_descriptors() && machine.fds().permit_sockets) {

		const auto real_fd = machine.fds().translate(vfd);
		alignas(16) char buffer[128];
		socklen_t addrlen = sizeof(buffer);

		int res = accept(real_fd, (struct sockaddr *)buffer, &addrlen);
		if (res >= 0) {
			// Assign and translate the new fd to virtual fd
			res = machine.fds().assign_socket(res);
			machine.copy_to_guest(g_addr, buffer, addrlen);
			machine.copy_to_guest(g_addrlen, &addrlen, sizeof(addrlen));
		}
		machine.set_result_or_error(res);
		return;
	}
	machine.set_result(-EBADF);
}

template <int W>
static void syscall_connect(Machine<W>& machine)
{
	const auto [vfd, g_addr, addrlen] =
		machine.template sysargs<int, address_type<W>, address_type<W>> ();
	alignas(16) std::array<char, 128> buffer;

	if (addrlen > buffer.size()) {
		machine.set_result(-ENOMEM);
		return;
	}
	int real_fd = -EBADF;

	if (machine.has_file_descriptors() && machine.fds().permit_sockets) {

		real_fd = machine.fds().translate(vfd);
		machine.copy_from_guest(buffer.data(), g_addr, addrlen);

#ifdef SOCKETCALL_VERBOSE
		print_address<W>(buffer, addrlen);
#endif

		const int res = connect(real_fd, (const struct sockaddr *)buffer.data(), addrlen);
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}

	SYSPRINT("SYSCALL connect, vfd: %d (real_fd: %d) addr: 0x%lX len: %zu = %ld\n",
		vfd, real_fd, (long)g_addr, (size_t)addrlen, (long)machine.return_value());
}

template <int W>
static void syscall_getsockname(Machine<W>& machine)
{
	const auto [vfd, g_addr, g_addrlen] =
		machine.template sysargs<int, address_type<W>, address_type<W>> ();

	if (machine.has_file_descriptors() && machine.fds().permit_sockets)
	{
		const auto real_fd = machine.fds().translate(vfd);

		struct sockaddr addr {};
		socklen_t addrlen = 0;
		int res = getsockname(real_fd, &addr, &addrlen);
		if (res == 0) {
			machine.copy_to_guest(g_addr, &addr, addrlen);
			machine.copy_to_guest(g_addrlen, &addrlen, sizeof(addrlen));
		}
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}

	SYSPRINT("SYSCALL getsockname, fd: %d addr: 0x%lX len: 0x%lX = %ld\n",
		vfd, (long)g_addr, (long)g_addrlen, (long)machine.return_value());
}

template <int W>
static void syscall_getpeername(Machine<W>& machine)
{
	const auto [vfd, g_addr, g_addrlen] =
		machine.template sysargs<int, address_type<W>, address_type<W>> ();

	if (machine.has_file_descriptors() && machine.fds().permit_sockets)
	{
		const auto real_fd = machine.fds().translate(vfd);

		struct sockaddr addr {};
		socklen_t addrlen = 0;
		int res = getpeername(real_fd, &addr, &addrlen);

		if (res == 0) {
			machine.copy_to_guest(g_addr, &addr, addrlen);
			machine.copy_to_guest(g_addrlen, &addrlen, sizeof(addrlen));
		}
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}

	SYSPRINT("SYSCALL getpeername, fd: %d addr: 0x%lX len: 0x%lX = %ld\n",
		vfd, (long)g_addr, (long)g_addrlen, (long)machine.return_value());
}

template <int W>
static void syscall_sendto(Machine<W>& machine)
{
	// ssize_t sendto(int vfd, const void *buf, size_t len, int flags,
	//		   const struct sockaddr *dest_addr, socklen_t addrlen);
	const auto [vfd, g_buf, buflen, flags, g_dest_addr, dest_addrlen] =
		machine.template sysargs<int, address_type<W>, address_type<W>, int, address_type<W>, unsigned>();
	int real_fd = -1;

	if (dest_addrlen > 128) {
		machine.set_result(-ENOMEM);
		return;
	}
	alignas(16) char dest_addr[128];
	machine.copy_from_guest(dest_addr, g_dest_addr, dest_addrlen);

	if (machine.has_file_descriptors() && machine.fds().permit_sockets) {

		real_fd = machine.fds().translate(vfd);

#ifdef __linux__
		// Gather up to 1MB of pages we can read into
		std::array<riscv::vBuffer, 256> buffers;
		const size_t buffer_cnt =
			machine.memory.gather_buffers_from_range(buffers.size(), buffers.data(), g_buf, buflen);

		struct msghdr msg;
		msg.msg_name = dest_addr;
		msg.msg_namelen = static_cast<socklen_t>(dest_addrlen);
		msg.msg_iov = (struct iovec *)buffers.data();
		msg.msg_iovlen = buffer_cnt;
		msg.msg_control = nullptr;
		msg.msg_controllen = 0;
		msg.msg_flags = 0;

		const ssize_t res = sendmsg(real_fd, &msg, flags);
#else
		// XXX: Write me
		(void)real_fd;
		const ssize_t res = -1;
#endif
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL sendto, fd: %d (real fd: %d) len: %ld flags: %#x = %ld\n",
			 vfd, real_fd, (long)buflen, flags, (long)machine.return_value());
}

template <int W>
static void syscall_recvfrom(Machine<W>& machine)
{
	// ssize_t recvfrom(int vfd, void *buf, size_t len, int flags,
	// 					struct sockaddr *src_addr, socklen_t *addrlen);
	const auto [vfd, g_buf, buflen, flags, g_src_addr, g_addrlen] =
		machine.template sysargs<int, address_type<W>, address_type<W>, int, address_type<W>, address_type<W>>();
	int real_fd = -1;

	if (machine.has_file_descriptors() && machine.fds().permit_sockets) {

		real_fd = machine.fds().translate(vfd);

#ifdef __linux__
		// Gather up to 1MB of pages we can read into
		std::array<riscv::vBuffer, 256> buffers;
		const size_t buffer_cnt =
			machine.memory.gather_writable_buffers_from_range(buffers.size(), buffers.data(), g_buf, buflen);

		alignas(16) char dest_addr[128];
		struct msghdr hdr;
		hdr.msg_name = dest_addr;
		hdr.msg_namelen = sizeof(dest_addr);
		hdr.msg_iov = (struct iovec *)buffers.data();
		hdr.msg_iovlen = buffer_cnt;
		hdr.msg_control = nullptr;
		hdr.msg_controllen = 0;
		hdr.msg_flags = 0;
	#if 0
		printf("recvfrom(buffers: %zu, total: %zu)\n", buffer_cnt, size_t(buflen));
	#endif

		const ssize_t res = recvmsg(real_fd, &hdr, flags);
		if (res >= 0) {
			if (g_src_addr != 0x0)
				machine.copy_to_guest(g_src_addr, hdr.msg_name, hdr.msg_namelen);
			if (g_addrlen != 0x0)
				machine.copy_to_guest(g_addrlen, &hdr.msg_namelen, sizeof(hdr.msg_namelen));
		}
#else
		// XXX: Write me
		(void)real_fd;
		const ssize_t res = -1;
#endif
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL recvfrom, fd: %d (real fd: %d) len: %ld flags: %#x = %ld\n",
			 vfd, real_fd, (long)buflen, flags, (long)machine.return_value());
}

template <int W>
static void syscall_recvmsg(Machine<W>& machine)
{
	// ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags)
	const auto [vfd, g_msg, flags] =
		machine.template sysargs<int, address_type<W>, int>();
	int real_fd = -1;

	if (machine.has_file_descriptors() && machine.fds().permit_sockets) {

		real_fd = machine.fds().translate(vfd);

#ifdef __linux__
		std::array<riscv::vBuffer, 256> buffers;
		std::array<guest_iovec<W>, 256> g_iov;
		guest_msghdr<W> msg;
		machine.copy_from_guest(&msg, g_msg, sizeof(msg));

		if (msg.msg_iovlen > g_iov.size()) {
			machine.set_result(-ENOMEM);
			return;
		}
		machine.copy_from_guest(g_iov.data(), msg.msg_iov, msg.msg_iovlen * sizeof(guest_iovec<W>));

		unsigned vec_cnt = 0;
		size_t total = 0;
		for (unsigned i = 0; i < msg.msg_iovlen; i++) {
			const address_type<W> g_buf = g_iov[i].iov_base;
			const address_type<W> g_len = g_iov[i].iov_len;
			vec_cnt +=
				machine.memory.gather_writable_buffers_from_range(buffers.size() - vec_cnt, &buffers[vec_cnt], g_buf, g_len);
			total += g_len;
		}
	#if 0
		printf("recvmsg(buffers: %u, total: %zu)\n", vec_cnt, total);
	#endif

		alignas(16) char dest_addr[128];
		struct msghdr hdr;
		hdr.msg_name = dest_addr;
		hdr.msg_namelen = sizeof(dest_addr);
		hdr.msg_iov = (struct iovec *)buffers.data();
		hdr.msg_iovlen = vec_cnt;
		hdr.msg_control = nullptr;
		hdr.msg_controllen = 0;
		hdr.msg_flags = msg.msg_flags;

		const ssize_t res = recvmsg(real_fd, &hdr, flags);
		if (res >= 0) {
			if (msg.msg_name != 0x0) {
				machine.copy_to_guest(msg.msg_name, hdr.msg_name, hdr.msg_namelen);
				msg.msg_namelen = hdr.msg_namelen;
			}
		}
#else
		// XXX: Write me
		(void)real_fd;
		const ssize_t res = -1;
#endif
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL recvmsg, fd: %d (real fd: %d) msg: 0x%lX flags: %#x = %ld\n",
			 vfd, real_fd, (long)g_msg, flags, (long)machine.return_value());
}

template <int W>
static void syscall_sendmmsg(Machine<W>& machine)
{
	// ssize_t sendmmsg(int vfd, struct mmsghdr *msgvec, unsigned int vlen, int flags);
	const auto [vfd, g_msgvec, veclen, flags] =
		machine.template sysargs<int, address_type<W>, unsigned, int>();
	int real_fd = -1;

	if (machine.has_file_descriptors() && machine.fds().permit_sockets) {

		real_fd = machine.fds().translate(vfd);

#ifdef __linux__
		std::array<struct mmsghdr, 128> msgvec;

		if (veclen > msgvec.size()) {
			machine.set_result(-ENOMEM);
			return;
		}
		machine.copy_from_guest(&msgvec[0], g_msgvec, veclen * sizeof(msgvec[0]));

		ssize_t finalcnt = 0;
		for (size_t i = 0; i < veclen; i++)
		{
			auto& entry = msgvec[i].msg_hdr;
			std::array<guest_iovec<W>, 128> g_iov;

			if (entry.msg_iovlen > g_iov.size()) {
				machine.set_result(-ENOMEM);
				return;
			}
			machine.copy_from_guest(&g_iov[0], (address_type<W>)uintptr_t(entry.msg_iov), entry.msg_iovlen * sizeof(guest_iovec<W>));

			std::array<riscv::vBuffer, 128> buffers;
			unsigned vec_cnt = 0;
			size_t total_bytes = 0;
			for (unsigned i = 0; i < entry.msg_iovlen; i++) {
				const address_type<W> g_buf = g_iov[i].iov_base;
				const address_type<W> g_len = g_iov[i].iov_len;
				vec_cnt +=
					machine.memory.gather_buffers_from_range(buffers.size() - vec_cnt, &buffers[vec_cnt], g_buf, g_len);
				total_bytes += g_len;
			}

		#ifdef SOCKETCALL_VERBOSE
			printf("SYSCALL -- Vec %zu: Buffers: %u  msg_name=%p namelen: %u\n",
				i, vec_cnt, entry.msg_name, entry.msg_namelen);
		#endif

			struct msghdr hdr;
			hdr.msg_name = nullptr;
			hdr.msg_namelen = 0;
			hdr.msg_iov = (struct iovec *)buffers.data();
			hdr.msg_iovlen = vec_cnt;
			hdr.msg_control = nullptr;
			hdr.msg_controllen = 0;
			hdr.msg_flags = entry.msg_flags;

			const ssize_t res = sendmsg(real_fd, &hdr, flags);
			if (res < 0) {
				finalcnt = res;
				msgvec[i].msg_len = 0;
				break;
			} else if (res > 0) {
				finalcnt += 1;
				msgvec[i].msg_len = res;
			}
		}
		if (finalcnt > 0) {
			machine.copy_to_guest(g_msgvec, &msgvec[0], finalcnt * sizeof(msgvec[0]));
		}
#else
		// XXX: Write me
		(void)real_fd;
		const ssize_t finalcnt = 0;
#endif
		machine.set_result_or_error(finalcnt);
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL sendmmsg, fd: %d (real fd: %d) msgvec: 0x%lX flags: %#x = %ld\n",
			 vfd, real_fd, (long)g_msgvec, flags, (long)machine.return_value());
}

template <int W>
static void syscall_setsockopt(Machine<W>& machine)
{
	const auto [vfd, level, optname, g_opt, optlen] =
		machine.template sysargs<int, int, int, address_type<W>, unsigned> ();

	if (optlen > 128) {
		machine.set_result(-ENOMEM);
		return;
	}

	if (machine.has_file_descriptors() && machine.fds().permit_sockets) {

		const auto real_fd = machine.fds().translate(vfd);
		alignas(8) char buffer[128];
		machine.copy_from_guest(buffer, g_opt, optlen);

		int res = setsockopt(real_fd, level, optname, buffer, optlen);
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL setsockopt, fd: %d level: %x optname: %#x len: %u = %ld\n",
		vfd, level, optname, optlen, (long)machine.return_value());
}

template <int W>
static void syscall_getsockopt(Machine<W>& machine)
{
	const auto [vfd, level, optname, g_opt, g_optlen] =
		machine.template sysargs<int, int, int, address_type<W>, address_type<W>> ();
	socklen_t optlen = 0;

	if (machine.has_file_descriptors() && machine.fds().permit_sockets)
	{
		const auto real_fd = machine.fds().translate(vfd);

		alignas(8) char buffer[128];
		optlen = std::min(sizeof(buffer), size_t(g_optlen));
		int res = getsockopt(real_fd, level, optname, buffer, &optlen);
		if (res == 0) {
			machine.copy_to_guest(g_optlen, &optlen, sizeof(optlen));
			machine.copy_to_guest(g_opt, buffer, optlen);
		}
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}

	SYSPRINT("SYSCALL getsockopt, fd: %d level: %x optname: %#x len: %ld/%ld = %ld\n",
			 vfd, level, optname, (long)optlen, (long)g_optlen, (long)machine.return_value());
}

template <int W>
void add_socket_syscalls(Machine<W>& machine)
{
	machine.install_syscall_handler(198, syscall_socket<W>);
	machine.install_syscall_handler(200, syscall_bind<W>);
	machine.install_syscall_handler(201, syscall_listen<W>);
	machine.install_syscall_handler(202, syscall_accept<W>);
	machine.install_syscall_handler(203, syscall_connect<W>);
	machine.install_syscall_handler(204, syscall_getsockname<W>);
	machine.install_syscall_handler(205, syscall_getpeername<W>);
	machine.install_syscall_handler(206, syscall_sendto<W>);
	machine.install_syscall_handler(207, syscall_recvfrom<W>);
	machine.install_syscall_handler(208, syscall_setsockopt<W>);
	machine.install_syscall_handler(209, syscall_getsockopt<W>);
	machine.install_syscall_handler(212, syscall_recvmsg<W>);
	machine.install_syscall_handler(269, syscall_sendmmsg<W>);
}

#ifdef RISCV_32I
template void add_socket_syscalls<4>(Machine<4>&);
#endif
#ifdef RISCV_64I
template void add_socket_syscalls<8>(Machine<8>&);
#endif

} // riscv
