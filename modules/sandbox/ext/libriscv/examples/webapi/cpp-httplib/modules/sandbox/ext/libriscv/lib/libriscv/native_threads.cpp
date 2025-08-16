#include "threads.hpp"
#include "native_heap.hpp"
#include <cassert>
#include <cstdio>

namespace riscv {
	static const uint32_t STACK_SIZE = 256 * 1024;

template <int W>
void Machine<W>::setup_native_threads(const size_t syscall_base)
{
	if (this->m_mt == nullptr)
		this->m_mt.reset(new MultiThreading<W>(*this));

	// Globally register a system call that clobbers all registers
	Machine<W>::register_clobbering_syscall(syscall_base + 0); // microclone
	Machine<W>::register_clobbering_syscall(syscall_base + 1); // exit
	Machine<W>::register_clobbering_syscall(syscall_base + 2); // yield
	Machine<W>::register_clobbering_syscall(syscall_base + 3); // yield_to
	Machine<W>::register_clobbering_syscall(syscall_base + 4); // block
	Machine<W>::register_clobbering_syscall(syscall_base + 5); // unblock
	Machine<W>::register_clobbering_syscall(syscall_base + 6); // unblock_thread
	Machine<W>::register_clobbering_syscall(syscall_base + 8); // clone threadcall

	// 500: microclone
	this->install_syscall_handler(syscall_base+0,
	[] (Machine<W>& machine) {
		const auto stack = (machine.template sysarg<address_type<W>> (0) & ~0xF);
		const auto  func = machine.template sysarg<address_type<W>> (1);
		const auto   tls = machine.template sysarg<address_type<W>> (2);
		const auto flags = machine.template sysarg<uint32_t> (3);
		const auto sbase = machine.template sysarg<address_type<W>> (4);
		const auto ssize = machine.template sysarg<address_type<W>> (5);
		//printf(">>> clone(func=0x%lX, stack=0x%lX, tls=0x%lX, stack base=0x%lX size=0x%lX)\n",
		//		(long)func, (long)stack, (long)tls, (long)sbase, (long)ssize);
		auto* thread = machine.threads().create(
			CHILD_SETTID | flags, tls, 0x0, stack, tls, sbase, ssize);
		// suspend and store return value for parent: child TID
		auto* parent = machine.threads().get_thread();
		parent->suspend(thread->tid);
		// activate and setup a function call
		thread->activate();
		// NOTE: have to start at DST-4 here!!!
		machine.setup_call(tls);
		machine.cpu.jump(func-4);
	});
	// exit
	this->install_syscall_handler(syscall_base+1,
	[] (Machine<W>& machine) {
		const int status = machine.template sysarg<int> (0);
		THPRINT(machine,
			">>> Exit on tid=%d, exit status = %d\n",
				machine.threads().get_tid(), (int) status);
		// Exit returns true if the program ended
		if (!machine.threads().get_thread()->exit()) {
			// Should be a new thread now
			return;
		}
		machine.stop();
		machine.set_result(status);
	});
	// sched_yield
	this->install_syscall_handler(syscall_base+2,
	[] (Machine<W>& machine) {
		// begone!
		machine.threads().suspend_and_yield();
	});
	// yield_to
	this->install_syscall_handler(syscall_base+3,
	[] (Machine<W>& machine) {
		machine.threads().yield_to(machine.template sysarg<uint32_t> (0));
	});
	// block (w/reason)
	this->install_syscall_handler(syscall_base+4,
	[] (Machine<W>& machine) {
		// begone!
		if (machine.threads().block(machine.template sysarg<int> (0), 0))
			return;
		// error, we didn't block
		machine.set_result(-1);
	});
	// unblock (w/reason)
	this->install_syscall_handler(syscall_base+5,
	[] (Machine<W>& machine) {
		if (!machine.threads().wakeup_blocked(64, machine.template sysarg<int> (0)))
			machine.set_result(-1);
	});
	// unblock thread
	this->install_syscall_handler(syscall_base+6,
	[] (Machine<W>& machine) {
		machine.threads().unblock(machine.template sysarg<int> (0));
	});

	// super fast "direct" threads
	// N+8: clone threadcall
	this->install_syscall_handler(syscall_base+8,
	[] (Machine<W>& machine) {
		const auto func  = machine.sysarg(0);
		const auto fini  = machine.sysarg(1);

		const auto tls = machine.arena().malloc(STACK_SIZE);
		if (UNLIKELY(tls == 0)) {
			THPRINT(machine,
				"Error: Thread stack allocation failed: %#x\n", tls);
			machine.set_result(-1);
			return;
		}
		const auto stack = ((tls + STACK_SIZE) & ~0xFLL);

		auto* thread = machine.threads().create(
			CHILD_SETTID, tls, 0x0, stack, tls, tls, STACK_SIZE);
		// suspend and store return value for parent: child TID
		auto* parent = machine.threads().get_thread();
		parent->suspend(thread->tid);
		// activate and setup a function call
		thread->activate();
		// exit into the exit function which frees the thread
		machine.cpu.reg(riscv::REG_RA) = fini;
		// geronimo!
		machine.cpu.jump(func - 4);
	});
	// N+9: exit threadcall
	this->install_syscall_handler(syscall_base+9,
	[] (Machine<W>& machine) {
		auto retval = machine.cpu.reg(riscv::REG_RETVAL);
		auto self = machine.cpu.reg(riscv::REG_TP);
		// TODO: check this return value
		machine.arena().free(self);
		// exit thread instead
		machine.threads().get_thread()->exit();
		// return value from exited thread
		machine.set_result(retval);
	});
}

#ifdef RISCV_32I
template void Machine<4>::setup_native_threads(const size_t);
#endif
#ifdef RISCV_64I
template void Machine<8>::setup_native_threads(const size_t);
#endif
} // riscv
