#include <sys/epoll.h>
#include <sys/eventfd.h>
//#define SYSPRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)

template <int W>
static void syscall_eventfd2(Machine<W>& machine)
{
	const auto initval = machine.template sysarg<int>(0);
	const auto flags = machine.template sysarg<int>(1);
	int real_fd = -EBADFD;

	if (machine.has_file_descriptors()) {
		real_fd = eventfd(initval, flags);
		if (real_fd > 0) {
			const int vfd = machine.fds().assign_file(real_fd);
			machine.set_result(vfd);
		} else {
			machine.set_result_or_error(real_fd);
		}
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL eventfd2(initval: %X flags: %#x real_fd: %d) = %d\n",
		initval, flags, real_fd, machine.template return_value<int>());
}

template <int W>
static void syscall_epoll_create(Machine<W>& machine)
{
	const auto flags = machine.template sysarg<int>(0);
	int real_fd = -EBADFD;

	if (machine.has_file_descriptors()) {
		real_fd = epoll_create1(flags);
		if (real_fd > 0) {
			const int vfd = machine.fds().assign_file(real_fd);
			machine.set_result(vfd);
		} else {
			machine.set_result_or_error(real_fd);
		}
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL epoll_create(real_fd: %d), flags: %#x = %d\n",
		real_fd, flags, machine.template return_value<int>());
}

template <int W>
static void syscall_epoll_ctl(Machine<W>& machine)
{
	// int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);
	const auto vepoll_fd = machine.template sysarg<int>(0);
	const auto op  = machine.template sysarg<int>(1);
	const auto vfd = machine.template sysarg<int>(2);
	const auto g_event = machine.sysarg(3);
	int real_fd = -EBADF;

	if (machine.has_file_descriptors()) {
		const int epoll_fd = machine.fds().translate(vepoll_fd);
		real_fd = machine.fds().translate(vfd);

		struct epoll_event event;
		machine.copy_from_guest(&event, g_event, sizeof(struct epoll_event));

		const int res = epoll_ctl(epoll_fd, op, real_fd, &event);
		machine.set_result_or_error(res);
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL epoll_ctl, epoll_fd: %d  op: %d vfd: %d (real_fd: %d)  event: 0x%lX => %d\n",
		vepoll_fd, op, vfd, real_fd, (long)g_event, (int)machine.return_value());
}

template <int W>
static void syscall_epoll_pwait(Machine<W>& machine)
{
	//  int epoll_pwait(int epfd, struct epoll_event *events,
	//  				int maxevents, int timeout,
	//  				const sigset_t *sigmask);
	const auto vepoll_fd = machine.template sysarg<int>(0);
	const auto g_events = machine.sysarg(1);
	auto maxevents = machine.template sysarg<int>(2);
	auto timeout = machine.template sysarg<int>(3);
	if (timeout < 0 || timeout > 1) timeout = 1;

	std::array<struct epoll_event, 4096> events;
	if (maxevents < 0 || maxevents > (int)events.size()) {
		SYSPRINT("WARNING: Too many epoll events for %d\n", vepoll_fd);
		maxevents = events.size();
	}
	int real_fd = -EBADF;

	if (machine.has_file_descriptors()) {
		real_fd = machine.fds().translate(vepoll_fd);

		const int res = epoll_wait(real_fd, events.data(), maxevents, timeout);
		if (res > 0) {
			machine.copy_to_guest(g_events, events.data(), res * sizeof(struct epoll_event));
			machine.set_result(res);
		} else if (res < 0 || timeout == 0) {
			machine.set_result_or_error(res);
		} else {
			// Finish up: Set -EINTR, then yield
			if (machine.threads().suspend_and_yield(-EINTR)) {
				SYSPRINT("SYSCALL epoll_pwait yielded...\n");
				return;
			}
		}
	} else {
		machine.set_result(-EBADF);
	}
	SYSPRINT("SYSCALL epoll_pwait, epoll_fd: %d (real_fd: %d), maxevents: %d timeout: %d = %ld\n",
		   vepoll_fd, real_fd, maxevents, timeout, (long)machine.return_value());
}
