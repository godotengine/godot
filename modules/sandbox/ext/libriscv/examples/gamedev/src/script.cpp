#include "script.hpp"
using gaddr_t = Script::gaddr_t;

#include <fstream> // Windows doesn't implement C getline()
#include <fmt/core.h>
#include <libriscv/native_heap.hpp>
static std::vector<uint8_t> load_file(const std::string& filename);
static const int HEAP_SYSCALLS_BASE	  = 490; // 490-494 (5)
static const int MEMORY_SYSCALLS_BASE = 495; // 495-509 (15)
static const std::vector<std::string> env = {
	"LC_CTYPE=C", "LC_ALL=C", "USER=groot"
};

Script::Script(
	std::shared_ptr<const std::vector<uint8_t>> binary, const std::string& name,
	const std::string& filename, void* userptr)
  : m_binary(binary),
    m_userptr(userptr), m_name(name),
	m_filename(filename)
{
	static bool init = false;
	if (!init)
	{
		init = true;
		Script::setup_syscall_interface();
	}
	this->reset();
	this->initialize();
}

Script::Script(
	const std::string& name, const std::string& filename, void* userptr)
  : Script(std::make_shared<const std::vector<uint8_t>> (load_file(filename)), name, filename, userptr)
{}

Script Script::clone(const std::string& name, void* userptr)
{
	return Script(this->m_binary, name, this->m_filename, userptr);
}

Script::~Script() {}

void Script::reset()
{
	// If the reset fails, this object is still valid:
	// m_machine.reset() will not happen if new machine_t fails
	try
	{
		// Create a new machine based on m_binary */
		riscv::MachineOptions<MARCH> options {
			.memory_max		  = MAX_MEMORY,
			.stack_size		  = STACK_SIZE,
			.use_memory_arena = true,
			.default_exit_function = "fast_exit",
		};
		m_machine = std::make_unique<machine_t> (*m_binary, options);

		// setup system calls and traps
		this->machine_setup();
		// setup program argv *after* setting new stack pointer
		machine().setup_linux({name()}, env);
	}
	catch (std::exception& e)
	{
		fmt::print(">>> Exception during initialization: {}\n", e.what());
		throw;
	}
}

void Script::initialize()
{
	// run through the initialization
	try
	{
		machine().simulate(MAX_BOOT_INSTR);
	}
	catch (riscv::MachineTimeoutException& me)
	{
		fmt::print(">>> Exception: Instruction limit reached on {}\n"
			"Instruction count: {}\n",
			name(), machine().max_instructions());
		throw;
	}
	catch (riscv::MachineException& me)
	{
		fmt::print(">>> Machine exception {}: {}\n"
			" (data: 0x{:x})\n",
			me.type(), me.what(), me.data());
		throw;
	}
	catch (std::exception& e)
	{
		fmt::print(">>> Exception: {}\n", e.what());
		throw;
	}

	fmt::print(">>> {} initialized.\n", name());
}

void Script::machine_setup()
{
	machine().set_userdata<Script>(this);
	machine().set_printer((machine_t::printer_func)[](
		const machine_t&, const char* p, size_t len) {
		fmt::print("{}", std::string_view {p, len});
	});
	machine().on_unhandled_csr = [](machine_t& machine, int csr, int, int)
	{
		auto& script = *machine.template get_userdata<Script>();
		fmt::print("{}: Unhandled CSR: {}\n", script.name(), csr);
	};
	machine().on_unhandled_syscall = [](machine_t& machine, size_t num)
	{
		auto& script = *machine.get_userdata<Script>();
		fmt::print("{}: Unhandled system call: {}\n", script.name(), num);
	};
	// Allocate heap area using mmap
	this->m_heap_area = machine().memory.mmap_allocate(MAX_HEAP);

	// Add POSIX system call interfaces (no filesystem or network access)
	machine().setup_linux_syscalls(false, false);
	machine().setup_posix_threads();
	// Add native system call interfaces
	machine().setup_native_heap(HEAP_SYSCALLS_BASE, heap_area(), MAX_HEAP);
	machine().setup_native_memory(MEMORY_SYSCALLS_BASE);
}

void Script::could_not_find(std::string_view func)
{
	fmt::print("Script::call(): Could not find: '{}' in '{}'\n", func, name());
}

void Script::handle_exception(gaddr_t address)
{
	auto callsite = machine().memory.lookup(address);
	fmt::print("[{}] Exception when calling:\n  {} (0x{:x})\nBacktrace:\n",
		name(), callsite.name, callsite.address);
	this->print_backtrace(address);

	try
	{
		throw; // re-throw
	}
	catch (const riscv::MachineTimeoutException& e)
	{
		this->handle_timeout(address);
		return; // NOTE: might wanna stay
	}
	catch (const riscv::MachineException& e)
	{
		fmt::print("Exception: {} (data: 0x{:x})\n>>> {}\n>>> Machine registers:\n[PC\t0x{:x}] {}\n",
			e.what(), e.data(), machine().cpu.current_instruction_to_string(),
			machine().cpu.pc(), machine().cpu.registers().to_string());
	}
	catch (const std::exception& e)
	{
		fmt::print("\nMessage: {}\n\n", e.what());
	}
	fmt::print("Program page: {}\n", machine().memory.get_page_info(machine().cpu.pc()));
	fmt::print("Stack page: {}\n", machine().memory.get_page_info(machine().cpu.reg(2)));
}

void Script::handle_timeout(gaddr_t address)
{
	auto callsite = machine().memory.lookup(address);
	fmt::print("Script::call: Max instructions reached when calling '{}' (0x{:x})\n",
		callsite.name, callsite.address);
}

void Script::max_depth_exceeded(gaddr_t address)
{
	auto callsite = machine().memory.lookup(address);
	fmt::print("Script::call(): Max call depth exceeded when calling '{}' (0x{:x})\n",
		callsite.name, callsite.address);
}

void Script::print_backtrace(const gaddr_t addr)
{
	machine().memory.print_backtrace(
		[](std::string_view line)
		{
			fmt::print("-> {}\n", line);
		});
	auto origin = machine().memory.lookup(addr);
	fmt::print("-> [-] 0x{:x} + 0x{:x}: {}\n", origin.address, origin.offset, origin.name);
}

void Script::print(std::string_view text)
{
	if (this->m_last_newline)
	{
		fmt::print("[{}] says: {}", name(), text);
	}
	else
	{
		fmt::print("{}", text);
	}
	this->m_last_newline = (text.back() == '\n');
}

gaddr_t Script::address_of(const std::string& name) const
{
	// We need to cache lookups because they are fairly expensive
	// Dynamic executables usually have a hash lookup table for symbols,
	// but no such thing for static executables. So, we compensate by
	// storing symbols in a local cache in order to reduce latencies.
	auto it = m_lookup_cache.find(name);
	if (it != m_lookup_cache.end())
		return it->second;

	const auto addr = machine().address_of(name.c_str());
	m_lookup_cache.try_emplace(name, addr);
	return addr;
}

std::string Script::symbol_name(gaddr_t address) const
{
	auto callsite = machine().memory.lookup(address);
	return callsite.name;
}

gaddr_t Script::guest_alloc(gaddr_t bytes)
{
	return machine().arena().malloc(bytes);
}

gaddr_t Script::guest_alloc_sequential(gaddr_t bytes)
{
	return machine().arena().seq_alloc_aligned(bytes, 8);
}

bool Script::guest_free(gaddr_t addr)
{
	return machine().arena().free(addr) == 0x0;
}

#include <unistd.h>

std::vector<uint8_t> load_file(const std::string& filename)
{
	size_t size = 0;
	FILE* f		= fopen(filename.c_str(), "rb");
	if (f == NULL)
		throw std::runtime_error("Could not open file: " + filename);

	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);

	std::vector<uint8_t> result(size);
	if (size != fread(result.data(), 1, size, f))
	{
		fclose(f);
		throw std::runtime_error("Error when reading from file: " + filename);
	}
	fclose(f);
	return result;
}
