#include <libriscv/machine.hpp>
#include <cstdio>
#define SYSPRINT(fmt, ...) /* fmt */

namespace riscv
{
	template <int W>
	static void syscall_stub_zero(Machine<W>& machine)
	{
		SYSPRINT("SYSCALL stubbed (zero): %d\n", (int)machine.cpu.reg(17));
		machine.set_result(0);
	}

	template <int W>
	static void syscall_stub_nosys(Machine<W>& machine)
	{
		SYSPRINT("SYSCALL stubbed (nosys): %d\n", (int)machine.cpu.reg(17));
		machine.set_result(-38); // ENOSYS
	}

	template <int W>
	static void syscall_ebreak(riscv::Machine<W>& machine)
	{
		printf("\n>>> EBREAK at %#lX\n", (long)machine.cpu.pc());
		throw MachineException(UNHANDLED_SYSCALL, "EBREAK instruction");
	}

	template<int W>
	static void syscall_write(Machine<W>& machine)
	{
		const int vfd = machine.template sysarg<int>(0);
		const auto address = machine.sysarg(1);
		const size_t len = machine.sysarg(2);
		SYSPRINT("SYSCALL write, fd: %d addr: 0x%lX, len: %zu\n",
				vfd, (long) address, len);
		// We only accept standard output pipes, for now :)
		if (vfd == 1 || vfd == 2) {
			// Zero-copy retrieval of buffers (64kb)
			riscv::vBuffer buffers[16];
			const size_t cnt =
				machine.memory.gather_buffers_from_range(16, buffers, address, len);
			for (size_t i = 0; i < cnt; i++) {
				machine.print(buffers[i].ptr, buffers[i].len);
			}
			machine.set_result(len);
			return;
		}
		machine.set_result(-EBADF);
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
	static void syscall_brk(Machine<W>& machine)
	{
		auto new_end = machine.sysarg(0);
		if (new_end > machine.memory.heap_address() + Memory<W>::BRK_MAX) {
			new_end = machine.memory.heap_address() + Memory<W>::BRK_MAX;
		} else if (new_end < machine.memory.heap_address()) {
			new_end = machine.memory.heap_address();
		}

		SYSPRINT("SYSCALL brk, new_end: 0x%lX\n", (long)new_end);
		machine.set_result(new_end);
	}

	template <int W>
	void Machine<W>::setup_minimal_syscalls()
	{
		install_syscall_handler(SYSCALL_EBREAK, syscall_ebreak<W>);
		install_syscall_handler(57, syscall_stub_zero<W>);  // close
		install_syscall_handler(62, syscall_stub_nosys<W>); // lseek
		install_syscall_handler(64, syscall_write<W>);
		install_syscall_handler(80, syscall_stub_nosys<W>); // fstat
		install_syscall_handler(93, syscall_exit<W>);
		install_syscall_handler(214, syscall_brk<W>);
	}

#ifdef RISCV_32I
	template void Machine<4>::setup_minimal_syscalls();
#endif
#ifdef RISCV_64I
	template void Machine<8>::setup_minimal_syscalls();
#endif
} // riscv
