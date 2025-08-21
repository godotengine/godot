#include "../machine.hpp"
#include "../decoder_cache.hpp"
#include "../internal_common.hpp"
#include "../riscvbase.hpp"
#include "../rv32i_instr.hpp"
#include "../threaded_bytecodes.hpp"
extern "C" {
	bool riscv64gb_accurate_dispatch(void* cpu, uint64_t pc, uint64_t icounter, uint64_t maxcounter);
	bool riscv64gb_inaccurate_dispatch(
		void* cpu, void* exec, void* decoder, uint64_t pc, uint64_t current_begin, uint64_t current_end);
	uint32_t riscv64gb_cpu_relative_icounter;
	uint32_t riscv64gb_cpu_relative_imaxcounter;
	void*    riscv64gb_cpu_syscall_array;
}

namespace riscv {

template <int W>
void CPU<W>::simulate_inaccurate(address_t pc) {
	if constexpr (W == 8) {
		static_assert(DecoderCache<W>::SHIFT == 2,
			"DecoderCache SHIFT must be 1 for assembly-based dispatch");
		machine().set_instruction_counter(0);
		machine().set_max_instructions(UINT64_MAX);
		// Set the relative instruction counter for the inaccurate dispatch
		riscv64gb_cpu_relative_icounter = (uintptr_t)&machine().get_counters().first - (uintptr_t)this;
		riscv64gb_cpu_relative_imaxcounter = (uintptr_t)&machine().get_counters().second - (uintptr_t)this;
		riscv64gb_cpu_syscall_array = (void*)&machine().syscall_handlers[0];

		DecodedExecuteSegment<W> *exec = this->m_exec;
		address_t current_begin = exec->exec_begin();
		address_t current_end = exec->exec_end();
		DecoderData<W> *exec_decoder = exec->decoder_cache();

		// We need an execute segment matching current PC
		if (UNLIKELY(!(pc >= current_begin && pc < current_end)))
		{
			auto new_values = this->next_execute_segment(pc);
			exec = new_values.exec;
			pc = new_values.pc;
			current_begin = exec->exec_begin();
			current_end = exec->exec_end();
			exec_decoder = exec->decoder_cache();
		}

		DecoderData<W> *decoder = &exec_decoder[pc >> DecoderCache<W>::SHIFT];
		pc += decoder->block_bytes();

		riscv64gb_inaccurate_dispatch(this, exec, decoder, pc, current_begin, current_end);

	} else {
		// For 32-bit and 128-bit, we don't have an inaccurate dispatch
		throw MachineException(FEATURE_DISABLED, "Inaccurate dispatch is not implemented for this CPU width");
	}
}

	INSTANTIATE_64_IF_ENABLED(CPU);
} // riscv
