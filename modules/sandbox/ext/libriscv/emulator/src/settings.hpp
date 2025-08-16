#pragma once
#include <string>
#include <unordered_set>
template <int W>
static std::vector<riscv::address_type<W>> load_jump_hints(const std::string& filename, bool verbose = false);
template <int W>
static void store_jump_hints(const std::string& filename, const std::vector<riscv::address_type<W>>& hints);

#if defined(EMULATOR_MODE_LINUX)
	static constexpr bool full_linux_guest = true;
#else
	static constexpr bool full_linux_guest = false;
#endif
#if defined(EMULATOR_MODE_NEWLIB)
	static constexpr bool newlib_mini_guest = true;
#else
	static constexpr bool newlib_mini_guest = false;
#endif
#if defined(EMULATOR_MODE_MICRO)
	static constexpr bool micro_guest = true;
#else
	static constexpr bool micro_guest = false;
#endif
