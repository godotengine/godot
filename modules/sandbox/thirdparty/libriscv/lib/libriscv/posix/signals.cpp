#include "signals.hpp"

#include "../machine.hpp"
#include "../internal_common.hpp"
#include "../threads.hpp"

namespace riscv {

template <int W>
Signals<W>::Signals() {}
template <int W>
Signals<W>::~Signals() {}

template <int W>
SignalAction<W>& Signals<W>::get(int sig) {
	if (sig > 0)
		return signals.at(sig-1);
	throw MachineException(ILLEGAL_OPERATION, "Signal 0 invoked");
}

template <int W>
void Signals<W>::enter(Machine<W>& machine, int sig)
{
	if (sig == 0) return;

	auto& sigact = signals.at(sig);
	if (sigact.altstack) {
		auto* thread = machine.threads().get_thread();
		// Change to alternate per-thread stack
		auto& stack = per_thread(thread->tid).stack;
		machine.cpu.reg(REG_SP) = stack.ss_sp + stack.ss_size;
	}
	// We have to jump to handler-4 because we are mid-instruction
	// WARNING: Assumption.
	machine.cpu.jump(sigact.handler - 4);
}

	INSTANTIATE_32_IF_ENABLED(Signals);
	INSTANTIATE_64_IF_ENABLED(Signals);
	INSTANTIATE_128_IF_ENABLED(Signals);
} // riscv
