//#define SYSPRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)

template <int W>
static void syscall_eventfd2(Machine<W>& machine)
{
	const auto initval = machine.template sysarg<int>(0);
	const auto flags = machine.template sysarg<int>(1);
	int real_fd = -1;

	if (machine.has_file_descriptors()) {
		machine.set_result(123456780);
	} else {
		machine.set_result(-1);
	}
	SYSPRINT("SYSCALL eventfd2(initval: %X flags: %#x real_fd: %d) = %d\n",
		initval, flags, real_fd, machine.template return_value<int>());
}

template <int W>
static void syscall_epoll_create(Machine<W>& machine)
{
	const auto flags = machine.template sysarg<int>(0);
	int real_fd = -1;

	if (machine.has_file_descriptors()) {
		machine.set_result(123456781);
	} else {
		machine.set_result(-1);
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
	int real_fd = -1;

	if (machine.has_file_descriptors()) {
		machine.set_result(0);
	} else {
		machine.set_result(-1);
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

	struct epoll_event {
		uint32_t events;
		union {
			void *ptr;
			int fd;
			uint32_t u32;
			uint64_t u64;
		} data;
	};
	std::array<struct epoll_event, 4096> events;
	if (maxevents < 0 || maxevents > (int)events.size()) {
		SYSPRINT("WARNING: Too many epoll events for %d\n", vepoll_fd);
		maxevents = events.size();
	}
	int real_fd = -1;

	if (machine.has_file_descriptors()) {

		// Finish up: Set -EINTR, then yield
		if (machine.threads().suspend_and_yield(-EINTR)) {
			SYSPRINT("SYSCALL epoll_pwait yielded...\n");
			return;
		}

	} else {
		machine.set_result(-1);
	}
	SYSPRINT("SYSCALL epoll_pwait, epoll_fd: %d (real_fd: %d), maxevents: %d timeout: %d = %ld\n",
		   vepoll_fd, real_fd, maxevents, timeout, (long)machine.return_value());
}
