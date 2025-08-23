#include "machine.hpp"
#include "decoder_cache.hpp"
#include "instruction_counter.hpp"
#include "internal_common.hpp"
#include "riscvbase.hpp"
#include "rv32i_instr.hpp"
#include "threaded_bytecodes.hpp"
//#define TIME_EXECUTION

namespace riscv
{
#ifdef TIME_EXECUTION
	static timespec time_now()
	{
		timespec t;
		clock_gettime(CLOCK_MONOTONIC, &t);
		return t;
	}
	static long nanodiff(timespec start_time, timespec end_time)
	{
		return (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
	}
#endif

	// A default empty execute segment used to enforce that the
	// current CPU execute segment is never null.
	template <int W>
	std::shared_ptr<DecodedExecuteSegment<W>>& CPU<W>::empty_execute_segment() noexcept {
		static std::shared_ptr<DecodedExecuteSegment<W>> empty_shared = std::make_shared<DecodedExecuteSegment<W>>(0, 0, 0, 0);
		return empty_shared;
	}

	// Instructions may be unaligned with C-extension
	// On amd64 we take the cost, because it's faster
	union UnderAlign32 {
		uint16_t data[2];
		operator uint32_t() {
			return data[0] | uint32_t(data[1]) << 16;
		}
	};

	template <int W>
	CPU<W>::CPU(Machine<W>& machine, const Machine<W>& other)
		: m_machine { machine }, m_exec(other.cpu.m_exec)
	{
		// Copy all registers except vectors
		// Users can still copy vector registers by assigning to registers().rvv().
		this->registers().copy_from(Registers<W>::Options::NoVectors, other.cpu.registers());
	}
	template <int W>
	void CPU<W>::reset()
	{
		this->m_regs = {};
		this->reset_stack_pointer();
		// We can't jump if there's been no ELF loader
		if (!current_execute_segment().empty()) {
			const auto initial_pc = machine().memory.start_address();
			// Check if the initial PC is executable, unless
			// the execute segment is marked as execute-only.
			if (!current_execute_segment().is_execute_only())
			{
				const auto& page =
					machine().memory.get_exec_pageno(initial_pc / riscv::Page::size());
				if (UNLIKELY(!page.attr.exec))
					trigger_exception(EXECUTION_SPACE_PROTECTION_FAULT, initial_pc);
			}
			// This function will (at most) validate the execute segment
			this->jump(initial_pc);
		}
	}

	template <int W>
	DecodedExecuteSegment<W>& CPU<W>::init_execute_area(const void* vdata, address_t begin, address_t vlength, bool is_likely_jit)
	{
		if (vlength < 4)
			trigger_exception(EXECUTION_SPACE_PROTECTION_FAULT, begin);
		// Create a new *non-initial* execute segment
		if (machine().has_options())
			this->m_exec = &machine().memory.create_execute_segment(
				machine().options(), vdata, begin, vlength, false, is_likely_jit);
		else
			this->m_exec = &machine().memory.create_execute_segment(
				MachineOptions<W>(), vdata, begin, vlength, false, is_likely_jit);
		return *this->m_exec;
	} // CPU::init_execute_area

	template<int W> RISCV_NOINLINE
	typename CPU<W>::NextExecuteReturn CPU<W>::next_execute_segment(address_t pc)
	{
		static constexpr int MAX_RESTARTS = 4;
		int restarts = 0;
restart_next_execute_segment:

		// Find previously decoded execute segment
		this->m_exec = machine().memory.exec_segment_for(pc).get();
		if (LIKELY(!this->m_exec->empty() && !this->m_exec->is_stale())) {
			return {this->m_exec, pc};
		}

		// We absolutely need to write PC here because even read-fault handlers
		// like get_pageno() slowpaths could be reading PC.
		this->registers().pc = pc;

		// Immediately look at the page in order to
		// verify execute and see if it has a trap handler
		address_t base_pageno = pc / Page::size();
		address_t end_pageno  = base_pageno + 1;

		// Check for +exec
		const auto& current_page =
			machine().memory.get_pageno(base_pageno);
		if (UNLIKELY(!current_page.attr.exec)) {
			this->m_fault(*this, current_page);
			pc = this->pc();

			if (UNLIKELY(++restarts == MAX_RESTARTS))
				trigger_exception(EXECUTION_LOOP_DETECTED, pc);

			goto restart_next_execute_segment;
		}

		// Check for trap
		if (UNLIKELY(current_page.has_trap()))
		{
			// We pass PC as offset
			current_page.trap(pc & (Page::size() - 1), TRAP_EXEC, pc);
			pc = this->pc();

			// If PC changed, we will restart the process
			if (pc / Page::size() != base_pageno)
			{
				if (UNLIKELY(++restarts == MAX_RESTARTS))
					trigger_exception(EXECUTION_LOOP_DETECTED, pc);

				goto restart_next_execute_segment;
			}
		}

		// Evict stale execute segments
		if (this->m_exec->is_stale()) {
			machine().memory.evict_execute_segment(*this->m_exec);
		}

		// Find decoded execute segment via override
		// If it returns empty, we build a new execute segment
		auto& next = this->m_override_exec(*this);
		if (!next.empty()) {
			this->m_exec = &next;
			return {this->m_exec, this->registers().pc};
		}

		// Find the earliest execute page in new segment
		const uint8_t* base_page_data = current_page.data();

		while (base_pageno > 0) {
			const auto& page =
				machine().memory.get_pageno(base_pageno-1);
			if (page.attr.exec) {
				base_pageno -= 1;
				base_page_data = page.data();
			} else break;
		}

		// Find the last execute page in segment
		const uint8_t* end_page_data = current_page.data();
		while (end_pageno != 0) {
			const auto& page =
				machine().memory.get_pageno(end_pageno);
			if (page.attr.exec) {
				end_pageno += 1;
				end_page_data = page.data();
			} else break;
		}

		if (UNLIKELY(end_pageno <= base_pageno))
			throw MachineException(INVALID_PROGRAM, "Failed to create execute segment");
		const size_t n_pages = end_pageno - base_pageno;
		end_page_data += Page::size();
		const bool sequential = end_page_data == base_page_data + n_pages * Page::size();
		// Check if it's likely a JIT-compiled area
		const bool is_likely_jit = current_page.attr.exec && current_page.attr.write;

		// Allocate full execute area
		if (!sequential) {
			std::unique_ptr<uint8_t[]> area(new uint8_t[n_pages * Page::size()]);
			// Copy from each individual page
			for (address_t p = base_pageno; p < end_pageno; p++) {
				// Cannot use get_exec_pageno here as we may need
				// access to read fault handler.
				auto& page = machine().memory.get_pageno(p);
				const size_t offset = (p - base_pageno) * Page::size();
				std::memcpy(area.get() + offset, page.data(), Page::size());
			}

			// Decode and store it for later
			return {&this->init_execute_area(area.get(), base_pageno * Page::size(), n_pages * Page::size(), is_likely_jit), pc};
		} else {
			// We can use the sequential execute segment directly
			return {&this->init_execute_area(base_page_data, base_pageno * Page::size(), n_pages * Page::size(), is_likely_jit), pc};
		}
	} // CPU::next_execute_segment

	template <int W> RISCV_NOINLINE RISCV_INTERNAL
	typename CPU<W>::format_t CPU<W>::read_next_instruction_slowpath() const
	{
		// Fallback: Read directly from page memory
		const auto pageno = this->pc() / address_t(Page::size());
		const auto& page = machine().memory.get_exec_pageno(pageno);
		if (UNLIKELY(!page.attr.exec)) {
			trigger_exception(EXECUTION_SPACE_PROTECTION_FAULT, this->pc());
		}
		const auto offset = this->pc() & (Page::size()-1);
		format_t instruction;

		if (LIKELY(offset <= Page::size()-4)) {
			instruction.whole = uint32_t(*(UnderAlign32 *)(page.data() + offset));
			return instruction;
		}
		// It's not possible to jump to a misaligned address,
		// so there is necessarily 16-bit left of the page now.
		instruction.whole = *(uint16_t*) (page.data() + offset);

		// If it's a 32-bit instruction at a page border, we need
		// to get the next page, and then read the upper half
		if (UNLIKELY(instruction.is_long()))
		{
			const auto& slow_page = machine().memory.get_exec_pageno(pageno+1);
			instruction.half[1] = *(uint16_t*) slow_page.data();
		}

		return instruction;
	}

	template <int W>
	bool CPU<W>::is_executable(address_t addr) const noexcept {
		return m_exec->is_within(addr);
	}

	template <int W>
	typename CPU<W>::format_t CPU<W>::read_next_instruction() const
	{
		if (LIKELY(this->is_executable(this->pc()))) {
			auto* exd = m_exec->exec_data(this->pc());
			return format_t { *(uint32_t*) exd };
		}

		return read_next_instruction_slowpath();
	}

	template <int W>
	static inline rv32i_instruction decode_safely(const uint8_t* exec_seg_data, address_type<W> pc)
	{
		// Instructions may be unaligned with C-extension
		// On amd64 we take the cost, because it's faster
#    if defined(RISCV_EXT_COMPRESSED) && !defined(__x86_64__)
		return rv32i_instruction { *(UnderAlign32*) &exec_seg_data[pc] };
#    else  // aligned/unaligned loads
		return rv32i_instruction { *(uint32_t*) &exec_seg_data[pc] };
#    endif // aligned/unaligned loads
	}

	template<int W> RISCV_HOT_PATH()
	void CPU<W>::simulate_precise()
	{
		// Decoded segments are always faster
		// So, always have at least the current segment
		if (!is_executable(this->pc())) {
			this->next_execute_segment(this->pc());
		}

		auto* exec = this->m_exec;
restart_precise_sim:
		auto* exec_seg_data = exec->exec_data();

		for (; machine().instruction_counter() < machine().max_instructions();
			machine().increment_counter(1)) {

			auto pc = this->pc();

			// TODO: This can me made much faster
			if (UNLIKELY(!exec->is_within(pc))) {
				// This will produce a sequential execute segment for the unknown area
				// If it is not executable, it will throw an execute space protection fault
				auto new_values = this->next_execute_segment(pc);
				exec = new_values.exec;
				pc   = new_values.pc;
				goto restart_precise_sim;
			}

			auto instruction = decode_safely<W>(exec_seg_data, pc);
			this->execute(instruction);

			// increment PC
			if constexpr (compressed_enabled)
				registers().pc += instruction.length();
			else
				registers().pc += 4;
		} // while not stopped

	} // CPU::simulate_precise

	template<int W>
	void CPU<W>::step_one(bool use_instruction_counter)
	{
		// Read, decode & execute instructions directly
		auto instruction = this->read_next_instruction();
		this->execute(instruction);

		if constexpr (compressed_enabled)
			registers().pc += instruction.length();
		else
			registers().pc += 4;

		machine().increment_counter(use_instruction_counter ? 1 : 0);
	}

	template<int W>
	address_type<W> CPU<W>::preempt_internal(Registers<W>& old_regs, bool Throw, bool store_regs, address_t pc, uint64_t max_instr)
	{
		auto& m = machine();
		const auto prev_max = m.max_instructions();
		try {
			// execute by extending the max instruction counter (resuming)
			// WARNING: Do not change this, as resumption is required in
			// order for sandbox integrity. Repeatedly invoking preemption
			// should lead to timeouts on either preempt() *or* the caller.
			m.simulate_with(
				m.instruction_counter() + max_instr, m.instruction_counter(), pc);
		} catch (...) {
			m.set_max_instructions(prev_max);
			if (store_regs) {
				this->registers() = old_regs;
			}
			if (Throw)
				throw; // Only rethrow if we're supposed to forward exceptions
		}
		// restore registers and return value
		m.set_max_instructions(prev_max);
		const auto retval = this->reg(REG_ARG0);
		if (store_regs) {
			this->registers() = old_regs;
		}
		return retval;
	}

	template <int W>
	DecoderData<W>& CPU<W>::create_block_ending_entry_at(DecodedExecuteSegment<W>& exec, address_t addr)
	{
		if (!exec.is_within(addr)) {
			throw MachineException(EXECUTION_SPACE_PROTECTION_FAULT,
				"Breakpoint address is not within the execute segment", addr);
		}

		auto* exec_decoder = exec.decoder_cache();
		auto* decoder_begin = &exec_decoder[exec.exec_begin() / DecoderCache<W>::DIVISOR];

		auto& cache_entry = exec_decoder[addr / DecoderCache<W>::DIVISOR];

		// The last instruction will be the current entry
		// Later instructions will work as normal
		// 1. Look back to find the beginning of the block
		auto* last    = &cache_entry;
		auto* current = &cache_entry;
		auto last_block_bytes = cache_entry.block_bytes();
		while (current > decoder_begin && (current-1)->block_bytes() > last_block_bytes) {
			current--;
			last_block_bytes = current->block_bytes();
		}

		// 2. Find the start address of the block
		const auto block_begin_addr = addr - (compressed_enabled ? 2 : 4) * (last - current);
		if (!exec.is_within(block_begin_addr)) {
			throw MachineException(INVALID_PROGRAM,
				"Breakpoint block was outside execute area", block_begin_addr);
		}

		// 3. Correct block_bytes() for all entries in the block
		auto patched_addr = block_begin_addr;
		for (auto* dd = current; dd < last; dd++) {
			// Get the patched decoder entry
			auto& p = exec_decoder[patched_addr / DecoderCache<W>::DIVISOR];
			p.idxend = last - dd;
		#ifdef RISCV_EXT_C
			p.icount = 0; // TODO: Implement C-ext icount for breakpoints
		#endif
			patched_addr += (compressed_enabled) ? 2 : 4;
		}
		// Check if the last address matches the breakpoint address
		if (patched_addr != addr) {
			throw MachineException(INVALID_PROGRAM,
				"Last instruction in breakpoint block was not aligned", patched_addr);
		}

		return cache_entry;
	}

	// Install an ebreak instruction at the given address
	template <int W>
	uint32_t CPU<W>::install_ebreak_for(DecodedExecuteSegment<W>& exec, address_t breakpoint_addr)
	{
		// Get a reference to the decoder cache
		auto& cache_entry = CPU<W>::create_block_ending_entry_at(exec, breakpoint_addr);
		const auto old_instruction = cache_entry.instr;

		// Install the new ebreak instruction at the breakpoint address
		rv32i_instruction new_instruction;
		new_instruction.Itype.opcode = 0b1110011; // SYSTEM
		new_instruction.Itype.rd = 0;
		new_instruction.Itype.funct3 = 0b000;
		new_instruction.Itype.rs1 = 0;
		new_instruction.Itype.imm = 1; // EBREAK
		cache_entry.instr = new_instruction.whole;
		cache_entry.set_bytecode(RV32I_BC_SYSTEM);
		cache_entry.idxend = 0;
	#ifdef RISCV_EXT_C
		cache_entry.icount = 0; // TODO: Implement C-ext icount for breakpoints
	#endif

		// Return the old instruction
		return old_instruction;
	}

	template <int W>
	uint32_t CPU<W>::install_ebreak_at(address_t addr)
	{
		return install_ebreak_for(*m_exec, addr);
	}

	template <int W>
	bool CPU<W>::create_fast_path_function(DecodedExecuteSegment<W>& exec, address_t block_pc)
	{
		// First, find the end of the block that either returns or stops (ignore traps)
		// 1. Return: JALR reg
		// 2. Stop: STOP
		if (!exec.is_within(block_pc)) {
			throw MachineException(EXECUTION_SPACE_PROTECTION_FAULT,
				"Function start address is not within the execute segment", block_pc);
		}

		auto* exec_decoder = exec.decoder_cache();
		// The beginning of the function:
		auto* cache_entry = &exec_decoder[block_pc / DecoderCache<W>::DIVISOR];

		const address_t current_end = exec.exec_end();
		while (block_pc < current_end)
		{
			// Move to the end of the block
			block_pc += cache_entry->block_bytes();
			cache_entry += cache_entry->block_bytes() / DecoderCache<W>::DIVISOR;
			// Check if we're still within the execute segment
			if (UNLIKELY(block_pc >= current_end)) {
				// TODO: Return false instead?
				throw MachineException(INVALID_PROGRAM,
					"Function block ended outside execute area", block_pc);
			}
			// Check if we're at the end of the function
			auto bytecode = cache_entry->get_bytecode();
			if (bytecode == RV32I_BC_JALR || bytecode == RV32I_BC_STOP) {
				const FasterItype instr { cache_entry->instr };

				if (bytecode == RV32I_BC_JALR) {
					// Check if it's a direct jump to REG_RA
					if (instr.rs2 == REG_RA && instr.rs1 == 0 && instr.imm == 0) {
						if (cache_entry->block_bytes() != 0)
							throw MachineException(INVALID_PROGRAM,
								"Function block ended but was not last instruction in block", block_pc);
						// We found the (potential) end of the function
						// Now rewrite it to a STOP instruction
						cache_entry->set_atomic_bytecode_and_handler(RV32I_BC_LIVEPATCH, 1);
						return true;
					} else {
						// Unconditional jump could be a tail call, in which
						// case we can't confidently optimize this function
						return false;
					}
				} else if (bytecode == RV32I_BC_STOP) {
					// It's already a fast-path function
					return true;
				}

				// Which instructions end the function?
			}

			cache_entry++;
			block_pc += (compressed_enabled) ? 2 : 4;
		}
		// Not able to find the end of the function
		return false;
	}

	template <int W>
	bool CPU<W>::create_fast_path_function(address_t addr)
	{
		DecodedExecuteSegment<W>* exec = machine().memory.exec_segment_for(addr).get();
		return create_fast_path_function(*exec, addr);
	}

	template<int W> RISCV_COLD_PATH()
	void CPU<W>::trigger_exception(int intr, address_t data)
	{
		switch (intr)
		{
		case INVALID_PROGRAM:
			throw MachineException(intr,
				"Machine not initialized", data);
		case ILLEGAL_OPCODE:
			throw MachineException(intr,
					"Illegal opcode executed", data);
		case ILLEGAL_OPERATION:
			throw MachineException(intr,
					"Illegal operation during instruction decoding", data);
		case PROTECTION_FAULT:
			throw MachineException(intr,
					"Protection fault", data);
		case EXECUTION_SPACE_PROTECTION_FAULT:
			throw MachineException(intr,
					"Execution space protection fault", data);
		case EXECUTION_LOOP_DETECTED:
			throw MachineException(intr,
					"Execution loop detected", data);
		case MISALIGNED_INSTRUCTION:
			// NOTE: only check for this when jumping or branching
			throw MachineException(intr,
					"Misaligned instruction executed", data);
		case INVALID_ALIGNMENT:
			throw MachineException(intr,
					"Invalid alignment for address", data);
		case UNIMPLEMENTED_INSTRUCTION:
			throw MachineException(intr,
					"Unimplemented instruction executed", data);
		case DEADLOCK_REACHED:
			throw MachineException(intr,
					"Atomics deadlock reached", data);
		case OUT_OF_MEMORY:
			throw MachineException(intr,
					"Out of memory", data);

		default:
			throw MachineException(UNKNOWN_EXCEPTION,
					"Unknown exception", intr);
		}
	}

	template <int W> RISCV_COLD_PATH()
	std::string CPU<W>::to_string(format_t bits) const
	{
		return to_string(bits, decode(bits));
	}

	template <int W> RISCV_COLD_PATH()
	std::string CPU<W>::current_instruction_to_string() const
	{
		format_t instruction;
		try {
			instruction = this->read_next_instruction();
		} catch (...) {
			instruction = format_t {};
		}
		return to_string(instruction, decode(instruction));
	}

	template <int W> RISCV_COLD_PATH()
	std::string Registers<W>::flp_to_string() const
	{
		char buffer[800];
		int  len = 0;
		for (int i = 0; i < 32; i++) {
			auto& src = this->getfl(i);
			const char T = (src.i32[1] == 0) ? 'S' : 'D';
			if constexpr (true) {
				double val = (src.i32[1] == 0) ? src.f32[0] : src.f64;
				len += snprintf(buffer+len, sizeof(buffer) - len,
						"[%s\t%c%+.2f] ", RISCV::flpname(i), T, val);
			} else {
				if (src.i32[1] == 0) {
					double val = src.f64;
					len += snprintf(buffer+len, sizeof(buffer) - len,
							"[%s\t%c0x%lX] ", RISCV::flpname(i), T, *(int64_t *)&val);
				} else {
					float val = src.f32[0];
					len += snprintf(buffer+len, sizeof(buffer) - len,
							"[%s\t%c0x%X] ", RISCV::flpname(i), T, *(int32_t *)&val);
				}
			}
			if (i % 5 == 4) {
				len += snprintf(buffer+len, sizeof(buffer)-len, "\n");
			}
		}
		len += snprintf(buffer+len, sizeof(buffer) - len,
				"[FFLAGS\t0x%X] ", m_fcsr.fflags);
		return std::string(buffer, len);
	}

	INSTANTIATE_32_IF_ENABLED(CPU);
	INSTANTIATE_32_IF_ENABLED(Registers);
	INSTANTIATE_64_IF_ENABLED(CPU);
	INSTANTIATE_64_IF_ENABLED(Registers);
	INSTANTIATE_128_IF_ENABLED(CPU);
	INSTANTIATE_128_IF_ENABLED(Registers);
}
