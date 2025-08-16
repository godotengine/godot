#include "machine.hpp"
#include "decoder_cache.hpp"
#include "instruction_counter.hpp"
#include "threaded_bytecodes.hpp"
#include "rv32i_instr.hpp"
#include "rvfd.hpp"
#ifdef RISCV_EXT_COMPRESSED
#include "rvc.hpp"
#endif
#ifdef RISCV_EXT_VECTOR
#include "rvv.hpp"
#endif

/**
 * This file is included by threaded_dispatch.cpp and bytecode_dispatch.cpp
 * It implements the logic for switch-based and threaded dispatch.
 * 
 * All dispatch modes share bytecode_impl.cpp
**/

namespace riscv
{
#define VIEW_INSTR() \
	auto instr = *(rv32i_instruction *)&decoder->instr;
#define VIEW_INSTR_AS(name, x) \
	auto &&name = *(x *)&decoder->instr;
#define NEXT_INSTR()                  \
	if constexpr (compressed_enabled) \
		decoder += 2;                 \
	else                              \
		decoder += 1;                 \
	EXECUTE_INSTR();
#define NEXT_C_INSTR() \
	decoder += 1;      \
	EXECUTE_INSTR();

#define NEXT_BLOCK(len, OF)                 \
	pc += len;                              \
	decoder += len >> DecoderCache<W>::SHIFT;              \
	if constexpr (FUZZING) /* Give OOB-aid to ASAN */      \
	decoder = &exec_decoder[pc >> DecoderCache<W>::SHIFT]; \
	if constexpr (OF) {						\
		if (UNLIKELY(counter.overflowed())) \
			goto check_jump;				\
	}										\
	pc += decoder->block_bytes();                            \
	counter.increment_counter(decoder->instruction_count()); \
	EXECUTE_INSTR();

#define SAFE_INSTR_NEXT(len)                  \
	pc += len;                                \
	decoder += len >> DecoderCache<W>::SHIFT; \
	counter.increment_counter(1);

#define NEXT_SEGMENT()                                       \
	decoder = &exec_decoder[pc >> DecoderCache<W>::SHIFT];  \
	pc += decoder->block_bytes();                            \
	counter.increment_counter(decoder->instruction_count()); \
	EXECUTE_INSTR();

#define PERFORM_BRANCH()                 \
	if constexpr (VERBOSE_JUMPS) fprintf(stderr, "Branch 0x%lX >= 0x%lX (decoder=%p)\n", long(pc), long(pc + fi.signed_imm()), decoder); \
	if (LIKELY(!counter.overflowed())) { \
		NEXT_BLOCK(fi.signed_imm(), false);     \
	}                                    \
	pc += fi.signed_imm();               \
	goto check_jump;

#define PERFORM_FORWARD_BRANCH()         \
	if constexpr (VERBOSE_JUMPS) fprintf(stderr, "Fw.Branch 0x%lX >= 0x%lX\n", long(pc), long(pc + fi.signed_imm())); \
	NEXT_BLOCK(fi.signed_imm(), false);

#define OVERFLOW_CHECKED_JUMP() \
	goto check_jump


template <int W> DISPATCH_ATTR
bool CPU<W>::simulate(address_t pc, uint64_t inscounter, uint64_t maxcounter)
{
	static constexpr uint32_t XLEN = W * 8;
	using addr_t  = address_type<W>;
	using saddr_t = signed_address_type<W>;

#ifdef DISPATCH_MODE_THREADED
#include "threaded_bytecode_array.hpp"
#endif

	DecodedExecuteSegment<W>* exec = this->m_exec;
	address_t current_begin = exec->exec_begin();
	address_t current_end   = exec->exec_end();

	DecoderData<W>* exec_decoder = exec->decoder_cache();
	DecoderData<W>* decoder;

	InstrCounter counter{inscounter, maxcounter};

	// We need an execute segment matching current PC
	if (UNLIKELY(!(pc >= current_begin && pc < current_end)))
		goto new_execute_segment;

#  ifdef RISCV_BINARY_TRANSLATION
	// There's a very high chance that the (first) instruction is a translated function
	decoder = &exec_decoder[pc >> DecoderCache<W>::SHIFT];
	if (LIKELY(decoder->get_bytecode() == RV32I_BC_TRANSLATOR))
		goto begin_translated_function;
#  endif

continue_segment:
	decoder = &exec_decoder[pc >> DecoderCache<W>::SHIFT];

	pc += decoder->block_bytes();
	counter.increment_counter(decoder->instruction_count());

#ifdef DISPATCH_MODE_SWITCH_BASED

while (true) {
	switch (decoder->get_bytecode()) {
	#define INSTRUCTION(bc, lbl) case bc:

#else
	goto *computed_opcode[decoder->get_bytecode()];
	#define INSTRUCTION(bc, lbl) lbl:

#endif

#define DECODER()   (*decoder)
#define CPU()       (*this)
#define REG(x)      registers().get()[x]
#define REGISTERS() registers()
#define VECTORS()   registers().rvv()
#define MACHINE()   machine()

	/** Instruction handlers **/

#include "bytecode_impl.cpp"

INSTRUCTION(RV32I_BC_SYSTEM, rv32i_system) {
	VIEW_INSTR();
	// Make the current PC visible
	REGISTERS().pc = pc;
	// Make the instruction counters visible
	counter.apply(MACHINE());
	// Invoke SYSTEM
	MACHINE().system(instr);
	// Restore counters
	counter.retrieve_counters(MACHINE());
	if (UNLIKELY(counter.overflowed() || pc != REGISTERS().pc))
	{
		pc = REGISTERS().pc;
		goto check_jump;
	}
	// Overflow-check, next block
	NEXT_BLOCK(4, true);
}

#ifdef RISCV_BINARY_TRANSLATION
INSTRUCTION(RV32I_BC_TRANSLATOR, translated_function) {
	counter.increment_counter(-1);
begin_translated_function:
	auto max = counter.max();
	auto cnt = counter.value();
retry_translated_function:
	// Invoke translated code
	auto bintr_results = 
		exec->unchecked_mapping_at(decoder->instr)(*this, cnt, max, pc);
	pc = REGISTERS().pc;
	cnt = bintr_results.counter;
	max = bintr_results.max_counter;
	if (LIKELY(cnt < max && (pc - current_begin < current_end - current_begin))) {
		decoder = &exec_decoder[pc >> DecoderCache<W>::SHIFT];
		if (decoder->get_bytecode() == RV32I_BC_TRANSLATOR) {
			goto retry_translated_function;
		}
		counter.set_counters(cnt, max);
		goto continue_segment;
	}
	counter.set_counters(cnt, max);
	goto check_jump;
}
#endif // RISCV_BINARY_TRANSLATION

INSTRUCTION(RV32I_BC_SYSCALL, rv32i_syscall) {
	// Make the current PC visible
	REGISTERS().pc = pc;
	// Make the instruction counter(s) visible
	counter.apply(MACHINE());
	// Invoke system call
	MACHINE().system_call(REG(REG_ECALL));
	// Restore counters
	counter.retrieve_counters(MACHINE());
	if (UNLIKELY(counter.overflowed() || pc != REGISTERS().pc))
	{
		// System calls are always full-length instructions
		if constexpr (VERBOSE_JUMPS) {
			if (pc != REGISTERS().pc)
			fprintf(stderr, "SYSCALL jump from 0x%lX to 0x%lX\n",
				long(pc), long(REGISTERS().pc + 4));
		}
		pc = REGISTERS().pc + 4;
		goto check_jump;
	}
	NEXT_BLOCK(4, false);
}

INSTRUCTION(RV32I_BC_STOP, rv32i_stop) {
	REGISTERS().pc = pc + 4;
	MACHINE().set_instruction_counter(counter.value());
	return true;
}

#ifdef DISPATCH_MODE_SWITCH_BASED
	default:
		goto execute_invalid;
	} // switch case
} // while loop

#endif

check_jump:
	if (UNLIKELY(counter.overflowed()))
		goto counter_overflow;

	if (LIKELY(pc - current_begin < current_end - current_begin))
		goto continue_segment;
	else
		goto new_execute_segment;

counter_overflow:
#ifdef RISCV_LIBTCC
	// We need to check if we have a current exception
	if (UNLIKELY(CPU().has_current_exception()))
		goto handle_rethrow_exception;
#endif

	registers().pc = pc;
	MACHINE().set_instruction_counter(counter.value());

	// Machine stopped normally?
	return counter.max() == 0;

	// Change to a new execute segment
new_execute_segment: {
		auto new_values = this->next_execute_segment(pc);
		exec = new_values.exec;
		pc   = new_values.pc;
		current_begin = exec->exec_begin();
		current_end   = exec->exec_end();
		exec_decoder  = exec->decoder_cache();
	}
	goto continue_segment;

execute_invalid:
	// Calculate the current PC from the decoder pointer
	pc = (decoder - exec_decoder) << DecoderCache<W>::SHIFT;
	// Check if the instruction is still invalid
	try {
		if (decoder->instr == 0 && MACHINE().memory.template read<uint16_t>(pc) != 0) {
			exec->set_stale(true);
			goto new_execute_segment;
		}
	} catch (...) {}
	MACHINE().set_instruction_counter(counter.value());
	registers().pc = pc;
	trigger_exception(ILLEGAL_OPCODE, decoder->instr);

#ifdef RISCV_LIBTCC
handle_rethrow_exception:
	// We have an exception, so we need to rethrow it
	const auto except = CPU().current_exception();
	CPU().clear_current_exception();
	std::rethrow_exception(except);
#endif

} // CPU::simulate_XXX()

} // riscv
