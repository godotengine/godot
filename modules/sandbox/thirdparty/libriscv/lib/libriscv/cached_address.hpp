#pragma once
#include "machine.hpp"

namespace riscv
{
	/* Resolve and remember the address of a named function */
	template <int W>
	struct CachedAddress {
		auto get(const Machine<W>& m, const std::string& func) const {
			if (addr) return addr;
			return (addr = m.address_of(func));
		}
		auto get(const Machine<W>& m, const char* func) const {
			if (addr) return addr;
			return (addr = m.address_of(func));
		}

	private:
		mutable address_type<W> addr = 0;
	};

}
