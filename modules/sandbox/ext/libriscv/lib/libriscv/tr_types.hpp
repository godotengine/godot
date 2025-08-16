#include "types.hpp"
#include "rv32i_instr.hpp"
#include <unordered_set>
#include <vector>

namespace riscv
{
	template <int W>
	struct TransInstr;

	template <int W>
	struct TransOutput
	{
		std::unordered_map<std::string, std::string> defines;
		timespec t0;
		std::shared_ptr<std::string> code;
		std::string footer;
		std::vector<TransMapping<W>> mappings;
	};

	template <int W>
	struct TransInfo
	{
		const std::vector<rv32i_instruction> instr;
		address_type<W> basepc;
		address_type<W> endpc;
		address_type<W> segment_basepc;
		address_type<W> segment_endpc;
		address_type<W> gp;
		bool is_libtcc;
		bool trace_instructions;
		bool ignore_instruction_limit;
		bool use_shared_execute_segments;
		bool use_register_caching;
		bool use_syscall_clobbering_optimization;
		bool use_automatic_nbit_address_space;
		bool unsafe_remove_checks;
		std::unordered_set<address_type<W>> jump_locations;
		std::unordered_map<address_type<W>, address_type<W>> single_return_locations;
		// Pointer to all the other blocks (including current)
		std::vector<TransInfo<W>>* blocks = nullptr;
		// Pointer to list of ebreak-locations
		const std::unordered_set<address_type<W>>* ebreak_locations = nullptr;

		std::unordered_set<address_type<W>>& global_jump_locations;

		uintptr_t arena_ptr;
		address_type<W> arena_roend;
		address_type<W> arena_size;
	};
}
