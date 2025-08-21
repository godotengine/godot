#include "common.hpp"
#include "internal_common.hpp"
#define DISPATCH_MODE_SWITCH_BASED
#define DISPATCH_ATTR RISCV_HOT_PATH()
#define DISPATCH_FUNC simulate_bytecode

#define EXECUTE_INSTR() \
	continue;
#define EXECUTE_CURRENT() \
	EXECUTE_INSTR()
#define UNUSED_FUNCTION() \
	RISCV_UNREACHABLE(); break;

namespace riscv
{
	static constexpr bool VERBOSE_JUMPS = riscv::verbose_branches_enabled;
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
	static constexpr bool FUZZING = true;
#else
	static constexpr bool FUZZING = false;
#endif
}

#include "cpu_dispatch.cpp"

#include "cpu_inaccurate_dispatch.cpp"

namespace riscv
{
	INSTANTIATE_32_IF_ENABLED(CPU);
	INSTANTIATE_64_IF_ENABLED(CPU);
	INSTANTIATE_128_IF_ENABLED(CPU);
} // riscv
