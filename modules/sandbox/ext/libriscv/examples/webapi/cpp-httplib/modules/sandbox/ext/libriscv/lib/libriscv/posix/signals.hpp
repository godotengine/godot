#pragma once
#include <map>
#include <set>
#include "../types.hpp"

namespace riscv {
template <int W> struct Machine;
template <int W> struct Registers;

template <int W>
struct SignalStack {
	address_type<W> ss_sp = 0x0;
	int             ss_flags = 0x0;
	address_type<W> ss_size = 0;
};

template <int W>
struct SignalAction {
	static constexpr address_type<W> SIG_UNSET = ~(address_type<W>)0x0;
	bool is_unset() const noexcept {
		return handler == 0x0 || handler == SIG_UNSET;
	}
	address_type<W> handler = SIG_UNSET;
	bool altstack = false;
	unsigned mask = 0x0;
};

template <int W>
struct SignalReturn {
	Registers<W> regs;
};

template <int W>
struct SignalPerThread {
	SignalStack<W> stack;
	SignalReturn<W> sigret;
};

template <int W>
struct Signals {
	SignalAction<W>& get(int sig);
	void enter(Machine<W>&, int sig);

	// TODO: Lock this in the future, for multiproessing
	auto& per_thread(int tid) { return m_per_thread[tid]; }

	Signals();
	~Signals();
private:
	std::array<SignalAction<W>, 64> signals {};
	std::map<int, SignalPerThread<W>> m_per_thread;
};

} // riscv
