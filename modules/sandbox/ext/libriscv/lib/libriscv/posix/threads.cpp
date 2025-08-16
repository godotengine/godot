#include "../threads.hpp"

namespace riscv {

template <int W>
static inline void futex_op(Machine<W>& machine,
	address_type<W> addr, int futex_op, int val, uint32_t val3)
{
	using address_t = address_type<W>;
	#define FUTEX_WAIT           0
	#define FUTEX_WAKE           1
	#define FUTEX_WAIT_BITSET	 9
	#define FUTEX_WAKE_BITSET	10

	THPRINT(machine, ">>> futex(0x%lX, op=%d (0x%X), val=%d val3=0x%X)\n",
		(long)addr, futex_op & 0xF, futex_op, val, val3);

	if ((futex_op & 0xF) == FUTEX_WAIT || (futex_op & 0xF) == FUTEX_WAIT_BITSET)
	{
		const bool is_bitset = (futex_op & 0xF) == FUTEX_WAIT_BITSET;
		if (machine.memory.template read<uint32_t> (addr) == (uint32_t)val) {
			THPRINT(machine,
				"FUTEX: Waiting (blocked)... uaddr=0x%lX val=%d, bitset=%d\n", (long)addr, val, is_bitset);
			if (machine.threads().block(0, addr, is_bitset ? val3 : 0x0)) {
				return;
			}
			//throw MachineException(DEADLOCK_REACHED, "FUTEX deadlock", addr);
			// This should never happen, but it does, and we have to unlock the futex and continue
			// in order to be able to proceed with the execution. TODO: Investigate why this happens.
			machine.memory.template write<address_t> (addr, 0);
			machine.set_result(0);
			return;
		}
		THPRINT(machine,
			"FUTEX: Wait condition EAGAIN... uaddr=0x%lX val=%d, bitset=%d\n", (long)addr, val, is_bitset);
		// This thread isn't blocked, but yielding may be necessary
		machine.threads().suspend_and_yield(-EAGAIN);
		return;
	} else if ((futex_op & 0xF) == FUTEX_WAKE || (futex_op & 0xF) == FUTEX_WAKE_BITSET) {
		const bool is_bitset = (futex_op & 0xF) == FUTEX_WAKE_BITSET;
		THPRINT(machine,
			"FUTEX: Waking %d others on 0x%lX, bitset=%d\n", val, (long)addr, is_bitset);
		// XXX: Guaranteed not to expire early when
		// timeout != 0x0.
		unsigned awakened = machine.threads().wakeup_blocked(val, addr, is_bitset ? val3 : ~0x0);
		machine.template set_result<unsigned>(awakened);
		THPRINT(machine,
			"FUTEX: Awakened: %u\n", awakened);
		return;
	}
	THPRINT(machine,
		"WARNING: Unhandled futex op: %X\n", futex_op);
	machine.set_result(-EINVAL);
}

template <int W>
void Machine<W>::setup_posix_threads()
{
	if (this->m_mt == nullptr)
		this->m_mt.reset(new MultiThreading<W>(*this));

	static constexpr int SYSCALL_CLONE        = 220;
	static constexpr int SYSCALL_CLONE3	      = 435;
	static constexpr int SYSCALL_SCHED_YIELD  = 124;
	static constexpr int SYSCALL_EXIT         = 93;
	static constexpr int SYSCALL_EXIT_GROUP   = 94;
	static constexpr int SYSCALL_FUTEX        = 98;
	static constexpr int SYSCALL_FUTEX_TIME64 = 422;
	static constexpr int SYSCALL_TKILL        = 130;
	static constexpr int SYSCALL_TGKILL       = 131;
	// There may be more, but these are known to clobber all registers
	register_clobbering_syscall(SYSCALL_CLONE);
	register_clobbering_syscall(SYSCALL_CLONE3);
	register_clobbering_syscall(SYSCALL_SCHED_YIELD);
	register_clobbering_syscall(SYSCALL_EXIT);
	register_clobbering_syscall(SYSCALL_EXIT_GROUP);
	register_clobbering_syscall(SYSCALL_FUTEX);
	register_clobbering_syscall(SYSCALL_FUTEX_TIME64);
	register_clobbering_syscall(SYSCALL_TKILL);
	register_clobbering_syscall(SYSCALL_TGKILL);

	// exit & exit_group
	this->install_syscall_handler(93,
	[] (Machine<W>& machine) {
		const uint32_t status = machine.template sysarg<uint32_t> (0);
		THPRINT(machine,
			">>> Exit on tid=%d, exit code = %d\n",
				machine.threads().get_tid(), (int) status);
		// Exit returns true if the program ended
		if (!machine.threads().get_thread()->exit()) {
			// Should be a new thread now
			return;
		}
		machine.stop();
		machine.set_result(status);
	});
	// exit_group
	this->install_syscall_handler(94, syscall_handlers.at(93));
	// set_tid_address
	this->install_syscall_handler(96,
	[] (Machine<W>& machine) {
		const int clear_tid = machine.template sysarg<address_type<W>> (0);
		// Without initialized threads, assume tid = 0
		if (machine.has_threads()) {
			machine.threads().get_thread()->clear_tid = clear_tid;
			machine.set_result(machine.threads().get_tid());
		} else {
			machine.set_result(0);
		}
		THPRINT(machine,
			">>> set_tid_address(0x%X) = %d\n", clear_tid, machine.return_value<int> ());
	});
	// set_robust_list
	this->install_syscall_handler(99,
	[] (Machine<W>& machine) {
		address_t addr = machine.template sysarg<address_type<W>> (0);
		THPRINT(machine,
			">>> set_robust_list(0x%lX) = 0\n", (long)addr);
		//machine.threads().get_thread()->robust_list_head = addr;
		(void)addr;
		machine.set_result(0);
	});
	// sched_yield
	this->install_syscall_handler(124,
	[] (Machine<W>& machine) {
		THPRINT(machine, ">>> sched_yield()\n");
		// begone!
		machine.threads().suspend_and_yield();
	});
	// tgkill
	this->install_syscall_handler(131,
	[] (Machine<W>& machine) {
		const int tid = machine.template sysarg<int> (1);
		const int sig = machine.template sysarg<int> (2);
		THPRINT(machine,
			">>> tgkill on tid=%d signal=%d\n", tid, sig);
		auto* thread = machine.threads().get_thread(tid);
		if (thread != nullptr) {
			// If the signal is unhandled, exit the thread
			if (sig != 0 && machine.sigaction(sig).is_unset()) {
				if (!thread->exit())
					return;
			} else {
				// Jump to signal handler and change to altstack, if set
				machine.signals().enter(machine, sig);
				THPRINT(machine,
					"<<< tgkill signal=%d jumping to 0x%lX (sp=0x%lX)\n",
					sig, (long)machine.sigaction(sig).handler, (long)machine.cpu.reg(REG_SP));
				return;
			}
		}
		machine.stop();
	});
	// gettid
	this->install_syscall_handler(178,
	[] (Machine<W>& machine) {
		THPRINT(machine,
			">>> gettid() = %d\n", machine.threads().get_tid());
		machine.set_result(machine.threads().get_tid());
	});
	// futex
	this->install_syscall_handler(98,
	[] (Machine<W>& machine) {
		const auto addr = machine.template sysarg<address_type<W>> (0);
		const int fx_op = machine.template sysarg<int> (1);
		const int   val = machine.template sysarg<int> (2);
		const uint32_t val3 = machine.template sysarg<uint32_t> (5);

		futex_op<W>(machine, addr, fx_op, val, val3);
	});
	// futex_time64
	this->install_syscall_handler(422,
	[] (Machine<W>& machine) {
		const auto addr = machine.template sysarg<address_type<W>> (0);
		const int fx_op = machine.template sysarg<int> (1);
		const int   val = machine.template sysarg<int> (2);
		const uint32_t val3 = machine.template sysarg<uint32_t> (5);

		futex_op<W>(machine, addr, fx_op, val, val3);
	});
	// clone
	this->install_syscall_handler(220,
	[] (Machine<W>& machine) {
		/* int clone(int (*fn)(void *arg), void *child_stack, int flags, void *arg,
					 void *parent_tidptr, void *tls, void *child_tidptr) */
		const int  flags = machine.template sysarg<int> (0);
		const auto stack = machine.template sysarg<address_type<W>> (1);
#ifdef THREADS_DEBUG
		const auto  func = machine.template sysarg<address_type<W>> (2);
		const auto  args = machine.template sysarg<address_type<W>> (3);
#endif
		const auto  ptid = machine.template sysarg<address_type<W>> (4);
		const auto   tls = machine.template sysarg<address_type<W>> (5);
		const auto  ctid = machine.template sysarg<address_type<W>> (6);
		auto* parent = machine.threads().get_thread();
		auto* thread = machine.threads().create(flags, ctid, ptid, stack, tls, 0, 0);
		THPRINT(machine,
			">>> clone(func=0x%lX, stack=0x%lX, flags=%x, args=0x%lX,"
				" parent=%p, ctid=0x%lX ptid=0x%lX, tls=0x%lX) = %d\n",
				(long)func, (long)stack, flags, (long)args, parent,
				(long)ctid, (long)ptid, (long)tls, thread->tid);
		// store return value for parent: child TID
		parent->suspend(thread->tid);
		// activate and return 0 for the child
		thread->activate();
		machine.set_result(0);
	});
	// clone3
	this->install_syscall_handler(435,
	[] (Machine<W>& machine) {
		/* int clone3(struct clone3_args*, size_t len) */
		static constexpr uint32_t SETTLS = 0x00080000;
		struct clone3_args {
			address_type<W> flags;
			address_type<W> pidfd;
			address_type<W> child_tid;
			address_type<W> parent_tid;
			address_type<W> exit_signal;
			address_type<W> stack;
			address_type<W> stack_size;
			address_type<W> tls;
			address_type<W> set_tid_array;
			address_type<W> set_tid_count;
			address_type<W> cgroup;
		};
		const auto [args, size] = machine.template sysargs<clone3_args, address_type<W>> ();
		if (size < sizeof(clone3_args)) {
			machine.set_result(-ENOSPC);
			return;
		}

		const int  flags = args.flags;
		const auto stack = args.stack + args.stack_size;
		const auto  ptid = args.parent_tid;
		const auto  ctid = args.child_tid;
		auto tls = args.tls;
		if ((args.flags & SETTLS) == 0) {
			tls = machine.cpu.reg(REG_TP);
		}

		auto* parent = machine.threads().get_thread();
		THPRINT(machine,
			">>> clone3(stack=0x%lX, flags=%x,"
				" parent=%p, ctid=0x%lX ptid=0x%lX, tls=0x%lX)\n",
				(long)stack, flags, parent, (long)ctid, (long)ptid, (long)tls);
		auto* thread = machine.threads().create(flags, ctid, ptid, stack, tls, 0, 0);

		if (args.set_tid_count > 0) {
			address_type<W> set_tid = 0;
			machine.copy_from_guest(&set_tid, args.set_tid_array, sizeof(set_tid));
			thread->clear_tid = set_tid;
		}

		// store return value for parent: child TID
		parent->suspend(thread->tid);
		// activate and return 0 for the child
		thread->activate();
		machine.set_result(0);
	});
	// prlimit64
	this->install_syscall_handler(261,
	[] (Machine<W>& machine) {
		const int resource = machine.template sysarg<int> (1);
		const auto old_addr = machine.template sysarg<address_type<W>> (3);
		struct {
			address_type<W> cur = 0;
			address_type<W> max = 0;
		} lim;
		constexpr int RISCV_RLIMIT_STACK  = 3;
		constexpr int RISCV_RLIMIT_NOFILE = 7;
		if (old_addr != 0 && resource == RISCV_RLIMIT_STACK) {
			lim.cur = 0x200000;
			lim.max = 0x200000;
			machine.copy_to_guest(old_addr, &lim, sizeof(lim));
			machine.set_result(0);
		} else if (old_addr != 0 && resource == RISCV_RLIMIT_NOFILE) {
			lim.cur = 1024;
			lim.max = 1024;
			machine.copy_to_guest(old_addr, &lim, sizeof(lim));
			machine.set_result(0);
		} else {
			machine.set_result(-EINVAL);
		}
		THPRINT(machine,
			">>> prlimit64(0x%X) = %d\n", resource, machine.return_value<int>());
	});
}

template <int W>
int Machine<W>::gettid() const noexcept
{
	if (m_mt) return m_mt->get_tid();
	return 0;
}

#ifdef RISCV_32I
template void Machine<4>::setup_posix_threads();
template int Machine<4>::gettid() const noexcept;
#endif
#ifdef RISCV_64I
template void Machine<8>::setup_posix_threads();
template int Machine<8>::gettid() const noexcept;
#endif
} // riscv
