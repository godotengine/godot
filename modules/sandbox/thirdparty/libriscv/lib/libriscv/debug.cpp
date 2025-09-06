#include "debug.hpp"

#include "decoder_cache.hpp"
#include "internal_common.hpp"
#include "rv32i_instr.hpp"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace riscv
{
static inline std::vector<std::string> split(const std::string& txt, char ch)
{
	size_t pos = txt.find(ch);
	size_t initialPos = 0;
	std::vector<std::string> strs;

	while (pos != std::string::npos)
	{
		strs.push_back(txt.substr(initialPos, pos - initialPos));
		initialPos = pos + 1;

		pos = txt.find(ch, initialPos);
	}

	// Add the last one
	strs.push_back(txt.substr(initialPos, std::min(pos, txt.size()) - initialPos + 1));
	return strs;
}
template <int W>
void DebugMachine<W>::dprintf(const char* fmt, ...) const
{
	char buffer[2048];

	va_list args;
	va_start(args, fmt);
	int len = vsnprintf(buffer, sizeof(buffer), fmt, args);
	va_end(args);

	if (len > 0) {
		this->debug_print(buffer, len);
	} else {
		throw MachineException(OUT_OF_MEMORY, "Debug print buffer too small");
	}
}

template <int W>
void DebugMachine<W>::print_help() const
{
	const char* help_text = R"V0G0N(
  usage: command [options]
	commands:
	  ?, help               Show this informational text
	  q, quit               Exit the interactive debugger
	  c, continue           Continue execution, disable stepping
	  s, step [steps=1]     Run [steps] instructions, then break
	  b, break [addr]       Breakpoint when PC == addr
	  b, break [name]       Resolve symbol to addr, use as breakpoint
	  watch [addr] (len=XL) Breakpoint on [addr] changing
	  clear                 Clear all breakpoints
	  bt, backtrace         Display primitive backtrace
	  a, addrof [name]      Resolve symbol name to address (or 0x0)
	  read [addr] (len=1)   Read from [addr] (len) bytes and print
	  write [addr] [value]  Write [value] to memory location [addr]
	  print [addr] [length] Print [addr] as a string of [length] bytes
	  ebreak                Trigger the ebreak handler
	  syscall [num]         Trigger specific system call handler
	  f                     Print FP-registers
	  v, verbose            Toggle verbose instruction output
	  vr, vregs             Toggle verbose register output
	  vf, vfpregs           Toggle verbose fp-register output
	  vj, vjumps            Toggle verbose jump output
)V0G0N";
	dprintf("%s\n", help_text);
}

template <int W>
bool DebugMachine<W>::execute_commands()
{
	auto& cpu = machine.cpu;

	this->dprintf("Enter = cont, help, quit: ");
	std::string text;
	while (true)
	{
		const int c = getchar(); // press any key
		if (c == '\n' || c < 0)
			break;
		else
			text.append(1, (char) c);
	}
	if (text.empty()) return false;
	std::vector<std::string> params = split(text, ' ');
	const auto& cmd = params[0];

	// continue
	if (cmd == "c" || cmd == "continue")
	{
		this->break_on_steps(0);
		return false;
	}
	// stepping
	if (cmd == "")
	{
		return false;
	}
	else if (cmd == "s" || cmd == "step")
	{
		this->verbose_instructions = true; // ???
		int steps = 1;
		if (params.size() > 1) steps = std::stoi(params[1]);
		this->dprintf("Pressing Enter will now execute %d steps\n", steps);
		this->break_on_steps(steps);
		return false;
	}
	// breaking
	else if (cmd == "b" || cmd == "break")
	{
		if (params.size() < 2)
		{
			this->dprintf(">>> Not enough parameters: break [addr]\n");
			return true;
		}
		const auto addr = machine.address_of(params[1]);
		if (addr != 0x0) {
			this->dprintf("Breakpoint on %s with address 0x%lX\n",
				params[1].c_str(), addr);
			this->breakpoint(addr);
		} else {
			unsigned long hex = std::strtoul(params[1].c_str(), 0, 16);
			this->dprintf("Breakpoint on address 0x%lX\n", hex);
			this->breakpoint(hex);
		}
		return true;
	}
	else if (cmd == "clear")
	{
		this->breakpoints().clear();
		return true;
	}
	else if (cmd == "bt" || cmd == "backtrace")
	{
		machine.memory.print_backtrace(
			[&] (std::string_view line) {
				this->dprintf("-> %.*s\n", (int)line.size(), line.begin());
			});
		return true;
	}
	else if (cmd == "watch")
	{
		if (params.size() < 1)
		{
			this->dprintf(">>> Not enough parameters: watch [addr]\n");
			return true;
		}
		const auto addr = machine.address_of(params[1]);
		if (addr != 0x0) {
			this->dprintf("Watchpoint on %s with address 0x%lX\n",
				params[1].c_str(), addr);
			this->watchpoint(addr, W);
		} else {
			unsigned long hex = std::strtoul(params[1].c_str(), 0, 16);
			this->dprintf("Watchpoint on address 0x%lX\n", hex);
			this->watchpoint(hex, W);
		}
		return true;
	}
	else if (cmd == "a" || cmd == "addrof")
	{
		if (params.size() < 2)
		{
			this->dprintf(">>> Not enough parameters: addrof [name]\n");
			return true;
		}
		const auto addr = machine.address_of(params[1]);
		this->dprintf("The address of %s is 0x%lX.%s\n",
			params[1].c_str(), addr, addr == 0x0 ? " (Likely not found)" : "");
		return true;
	}
	// print registers
	else if (cmd == "f")
	{
		this->dprintf("%s\n", cpu.registers().flp_to_string().c_str());
		return true;
	}
	// verbose instructions
	else if (cmd == "v" || cmd == "verbose")
	{
		bool& v = this->verbose_instructions;
		v = !v;
		this->dprintf("Verbose instructions are now %s\n", v ? "ON" : "OFF");
		return true;
	}
	else if (cmd == "vr" || cmd == "vregs")
	{
		bool& v = this->verbose_registers;
		v = !v;
		this->dprintf("Verbose registers are now %s\n", v ? "ON" : "OFF");
		return true;
	}
	else if (cmd == "vf" || cmd == "vfpregs")
	{
		bool& v = this->verbose_fp_registers;
		v = !v;
		this->dprintf("Verbose FP-registers are now %s\n", v ? "ON" : "OFF");
		return true;
	}
	else if (cmd == "vj" || cmd == "vjumps")
	{
		bool& v = this->verbose_jumps;
		v = !v;
		this->dprintf("Verbose jumps are now %s\n", v ? "ON" : "OFF");
		return true;
	}
	else if (cmd == "r" || cmd == "run")
	{
		this->verbose_instructions = false;
		this->break_on_steps(0);
		return false;
	}
	else if (cmd == "q" || cmd == "quit" || cmd == "exit")
	{
		machine.stop();
		return false;
	}
	// read 0xAddr size
	else if (cmd == "lw" || cmd == "read")
	{
		if (params.size() < 2)
		{
			this->dprintf(">>> Not enough parameters: read [addr]\n");
			return true;
		}
		unsigned long addr = std::strtoul(params[1].c_str(), 0, 16);
		auto value = machine.memory.template read<uint32_t>(addr);
		this->dprintf("0x%lX: 0x%X\n", addr, value);
		return true;
	}
	// write 0xAddr value
	else if (cmd == "sw" || cmd == "write")
	{
		if (params.size() < 3)
		{
			this->dprintf(">>> Not enough parameters: write [addr] [value]\n");
			return true;
		}
		unsigned long hex = std::strtoul(params[1].c_str(), 0, 16);
		int value = std::stoi(params[2]) & 0xff;
		this->dprintf("0x%04lx -> 0x%02x\n", hex, value);
		machine.memory.template write<uint32_t>(hex, value);
		return true;
	}
	// print 0xAddr size
	else if (cmd == "print")
	{
		if (params.size() < 3)
		{
			this->dprintf(">>> Not enough parameters: print addr length\n");
			return true;
		}
		uint32_t src = std::strtoul(params[1].c_str(), 0, 16);
		int bytes = std::stoi(params[2]);
		std::unique_ptr<char[]> buffer(new char[bytes]);
		machine.memory.memcpy_out(buffer.get(), src, bytes);
		this->dprintf("0x%X: %.*s\n", src, bytes, buffer.get());
		return true;
	}
	else if (cmd == "ebreak")
	{
		machine.system_call(SYSCALL_EBREAK);
		return true;
	}
	else if (cmd == "syscall")
	{
		int num = 0;
		if (params.size() > 1) num = std::stoi(params[1]);
		this->dprintf("Triggering system call %d\n", num);
		machine.system_call(num);
		return true;
	}
	else if (cmd == "help" || cmd == "?")
	{
		this->print_help();
		return true;
	}
	else
	{
		this->dprintf(">>> Unknown command: '%s'\n", cmd.c_str());
		this->print_help();
		return true;
	}
	return false;
}

template<int W>
void DebugMachine<W>::print(const std::string& label, address_type<W> override_pc)
{
	auto& cpu = machine.cpu;
	auto old_pc = cpu.pc();
	try {
		if (override_pc != 0x0)
			cpu.aligned_jump(override_pc);

		const auto instruction = cpu.read_next_instruction();
		const auto& handler = cpu.decode(instruction);
		const auto string = cpu.to_string(instruction, handler);
		this->dprintf("\n>>> %s \t%s\n\n", label.c_str(), string.c_str());
	} catch (const std::exception& e) {
		this->dprintf("\n>>> %s \tError reading instruction: %s\n\n", label.c_str(), e.what());
	}
	cpu.aligned_jump(old_pc);
	// CPU registers
	this->dprintf("%s", cpu.registers().to_string().c_str());
	// Memory subsystem
	this->dprintf("[MEM PAGES     %8zu]\n", machine.memory.pages_active());
	// Floating-point registers
	if (this->verbose_fp_registers) {
		this->dprintf("%s", cpu.registers().flp_to_string().c_str());
	}
}

template<int W>
void DebugMachine<W>::print_and_pause()
{
	this->print();

	while (execute_commands())
		;
} // print_and_pause(...)

template<int W>
bool DebugMachine<W>::break_time() const
{
	if (UNLIKELY(m_break_steps_cnt != 0))
	{
		m_break_steps--;
		if (m_break_steps <= 0)
		{
			m_break_steps = m_break_steps_cnt;
			return true;
		}
	}
	return false;
}

template<int W>
void DebugMachine<W>::break_on_steps(int steps)
{
	assert(steps >= 0);
	this->m_break_steps_cnt = steps;
	this->m_break_steps = steps;
}

template<int W>
void DebugMachine<W>::break_checks()
{
	if (UNLIKELY(this->break_time()))
	{
		// pause for each instruction
		this->print_and_pause();
	}
	if (UNLIKELY(!m_breakpoints.empty()))
	{
		// look for breakpoints
		auto it = m_breakpoints.find(machine.cpu.pc());
		if (it != m_breakpoints.end())
		{
			it->second(*this);
		}
	}
	if (UNLIKELY(!m_watchpoints.empty()))
	{
		for (auto& wp : m_watchpoints) {
			/* TODO: Only run watchpoint on LOAD STORE instructions */
			address_type<W> new_value = 0;
			switch (wp.len) {
			case 1:
				new_value = machine.memory.template read<uint8_t> (wp.addr);
				break;
			case 2:
				new_value = machine.memory.template read<uint16_t> (wp.addr);
				break;
			case 4:
				new_value = machine.memory.template read<uint32_t> (wp.addr);
				break;
			case 8:
				new_value = machine.memory.template read<uint64_t> (wp.addr);
				break;
			}
			if (wp.last_value != new_value) {
				wp.callback(*this);
			}
			wp.last_value = new_value;
		}
	}
}

template<int W>
void DebugMachine<W>::register_debug_logging() const
{
	std::string regs = "\n";
	regs += machine.cpu.registers().to_string();
	regs += "\n\n";
	this->debug_print(regs.data(), regs.size());
	if (UNLIKELY(this->verbose_fp_registers)) {
		std::string fp_regs = machine.cpu.registers().flp_to_string();
		fp_regs += "\n";
		this->debug_print(fp_regs.data(), fp_regs.size());
	}
}

// Instructions may be unaligned with C-extension
// On amd64 we take the cost, because it's faster
union UnderAlign32
{
	uint16_t data[2];
	operator uint32_t()
	{
		return data[0] | uint32_t(data[1]) << 16;
	}
};

template<int W>
void DebugMachine<W>::simulate(std::function<void(DebugMachine<W>&)> callback, uint64_t imax)
{
	auto& cpu = machine.cpu;
	address_type<W> pc = cpu.pc();
	DecodedExecuteSegment<W>* exec;
	auto new_values = cpu.next_execute_segment(pc);
	exec = new_values.exec;
	pc   = new_values.pc;
	auto* exec_seg_data = exec->exec_data();
	std::unordered_map<address_type<W>, std::string> backtrace_lookup;

	// Calculate the instruction limit
	if (imax != UINT64_MAX)
		machine.set_max_instructions(machine.instruction_counter() + imax);
	else
		machine.set_max_instructions(UINT64_MAX);

	for (; machine.instruction_counter() < machine.max_instructions();
		machine.increment_counter(1)) {

		this->break_checks();

		// Callback that lets you break on custom conditions
		if (callback)
			callback(*this);

		pc = cpu.pc();

		// NOTE: Break checks can change PC, full read
		if (UNLIKELY(!exec->is_within(pc)))
		{
			// This will produce a sequential execute segment for the unknown area
			// If it is not executable, it will throw an execute space protection fault
			auto updated_values = cpu.next_execute_segment(pc);
			exec = updated_values.exec;
			pc   = updated_values.pc;
			exec_seg_data = exec->exec_data();
		}

		// Instructions may be unaligned with C-extension
		const rv32i_instruction instruction =
			rv32i_instruction { *(UnderAlign32*) &exec_seg_data[pc] };
		if (this->verbose_instructions) {
			auto it = backtrace_lookup.find(pc);
			if (it == backtrace_lookup.end()) {
				std::string string = cpu.to_string(instruction) + " ";
				if (string.size() < 48)
					string.resize(48, ' ');

				machine.memory.print_backtrace([&] (auto view) {
					string.append(view);
				}, false);
				string.append("\n");
				// Print and move-insert at the same time
				machine.print(string.c_str(), string.size());
				backtrace_lookup.insert_or_assign(pc, std::move(string));
			} else {
				const std::string& bt = it->second;
				machine.print(bt.c_str(), bt.size());
			}
		}

		// Avoid decoder cache when debugging, as it may contain custom handlers
		cpu.execute(instruction);

		if (UNLIKELY(this->verbose_registers)) {
			this->register_debug_logging();
		}

		// increment PC
		if constexpr (compressed_enabled)
			cpu.registers().pc += instruction.length();
		else
			cpu.registers().pc += 4;
	} // while not stopped

} // DebugMachine::simulate

template <int W>
DebugMachine<W>::DebugMachine(Machine<W>& m) : machine(m) {
	m_debug_printer = [](const Machine<W>&, const char* buffer, size_t len) {
		fprintf(stderr, "[DebugMachine] %.*s\n", (int)len, buffer);
	};
}

template <int W>
void DebugMachine<W>::debug_print(const char* buffer, size_t len) const
{
	if (m_debug_printer)
		m_debug_printer(machine, buffer, len);
}

template<int W>
void DebugMachine<W>::simulate(uint64_t imax)
{
	this->simulate(nullptr, imax);
}

	INSTANTIATE_32_IF_ENABLED(DebugMachine);
	INSTANTIATE_64_IF_ENABLED(DebugMachine);
	INSTANTIATE_128_IF_ENABLED(DebugMachine);
} // riscv
