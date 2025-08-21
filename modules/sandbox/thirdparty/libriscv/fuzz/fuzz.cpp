#include <libriscv/machine.hpp>
#include "helpers.cpp"

static const std::vector<uint8_t> empty;
static constexpr int W = RISCV_ARCH;
static constexpr uint32_t MAX_CYCLES = 5'000;
static constexpr bool FUZZ_SYSTEM_CALLS = true;

// In order to be able to inspect a coredump we want to
// crash on every ASAN error.
extern "C" void __asan_on_error()
{
	abort();
}
extern "C" void __msan_on_error()
{
	abort();
}

/**
 * Fuzzing the instruction set only is very fast, and sometimes enough if
 * new instructions were added, and no other parts of the code has been
 * touched.
 *
**/
static void fuzz_instruction_set(const uint8_t* data, size_t len)
{
	constexpr uint32_t S = 0x1000;
	constexpr uint32_t V = 0x2000;

	try
	{
		riscv::Machine<W> machine { empty, {} };
		machine.memory.set_page_attr(S, 0x1000, {.read = true, .write = true});
		machine.memory.set_page_attr(V, 0x1000, {.read = true, .exec = true});
		machine.on_unhandled_syscall = [] (auto&, size_t) {};
		machine.cpu.init_execute_area(data, V, len);
		//machine.cpu.reg(riscv::REG_RA) = 0xffffffff;
		//machine.cpu.reg(riscv::REG_SP) = 0x1;
		machine.cpu.jump(V);
		// Let's avoid loops
		machine.simulate(MAX_CYCLES);
	}
	catch (const std::exception &e)
	{
		//printf(">>> Exception: %s\n", e.what());
	}
}

/**
 * It is tempting to re-use an existing machine when fuzzing, in order
 * to drastically increase the executions/sec, but if you ever find a
 * legitimate crash that way, you can almost never reproduce it through
 * just the payload.
 *
 * So, instead we create everything from scratch every time, and should it
 * ever crash, we will for sure be able to reproduce it in a unit test after.
 *
**/
static void fuzz_elf_loader(const uint8_t* data, size_t len)
{
	using namespace riscv;
	const std::string_view bin {(const char*) data, len};
	try {
		const MachineOptions<W> options {
			.allow_write_exec_segment = true,
			.use_memory_arena = false
		};
		// Instantiating the machine will fuzz the ELF loader,
		// and the decoder cache creation.
		Machine<W> machine { bin, options };
		machine.on_unhandled_syscall = [] (auto&, size_t) {};
		// Fuzzing the linux system calls with no filesystem access
		// and no socket access.
		if constexpr (FUZZ_SYSTEM_CALLS && W != 16) {
			// The fuzzer occasionally tries to invoke write/writev
			machine.set_printer([] (auto&, const char *, size_t) {});
			//machine.set_stdin([] (auto&, const char *, size_t) {});
			machine.setup_linux_syscalls(false, false);
			machine.setup_linux({"program"}, {"LC_ALL=C"});
		}
		machine.simulate(MAX_CYCLES);
	} catch (const std::exception& e) {
		//printf(">>> Exception: %s\n", e.what());
	}
}

/**
 * This mode attempts to fuzz the native system calls that ship
 * with libriscv. These syscalls drastically improve certain
 * common libc functions. Guests are free to use them, or simply
 * ignore them.
**/
static void fuzz_native_syscalls(const uint8_t* data, size_t len)
{
	static constexpr uint32_t S = 0x1000;
	static constexpr uint32_t V = 0x2000;
	static constexpr uint64_t MAX_CYCLES = 10'000ull;

	if (len < 1)
		return;

	// Create empty machine
	const riscv::MachineOptions<W> options {
		.allow_write_exec_segment = true,
		.use_memory_arena = false
	};
	riscv::Machine<W> machine { empty, options };
	machine.on_unhandled_syscall = [] (auto&, size_t) {};
	// The fuzzer occasionally tries to invoke write/writev
	machine.set_printer([] (auto&, const char *, size_t) {});

	const uint8_t syscall_id = data[0];
	try {
		// Allocate a custom guest heap
		const auto heap = machine.memory.mmap_allocate(2ull << 20);
		machine.setup_native_memory(100);
		machine.setup_native_heap(200, heap, 2ull << 20);
		machine.setup_native_threads(300);
		machine.setup_argv({"program"}, {"LC_ALL=C"});

		machine.cpu.init_execute_area(&data[1], V, len - 1);
		machine.cpu.jump(V);

		machine.simulate(MAX_CYCLES);
	} catch (const std::exception&) {
		//printf(">>> Exception: %s\n", e.what());
	}
	// Accelerate syscall fuzzing
	try {
		machine.system_call(syscall_id);
	} catch (...) {
	}
}

/**
 * This mode attempts to fuzz the system call helpers that
 * are designed to make it easy and safe to implement host-side
 * system calls in libriscv.
**/
static void fuzz_syscall_helpers(const uint8_t* data, size_t len)
{
	static constexpr uint32_t S = 0x1000;
	static constexpr uint32_t V = 0x2000;
	static constexpr uint64_t MAX_CYCLES = 10'000ull;

	// Create empty machine
	const riscv::MachineOptions<W> options {
		.allow_write_exec_segment = true,
		.use_memory_arena = false
	};
	riscv::Machine<W> machine { empty, options };
	machine.on_unhandled_syscall = [] (auto&, size_t) {};
	// The fuzzer occasionally tries to invoke write/writev
	machine.set_printer([] (auto&, const char *, size_t) {});

	const uint8_t syscall_id = data[0];
	try {
		// Allocate a custom guest heap
		const auto heap = machine.memory.mmap_allocate(2ull << 20);

		machine.install_syscall_handler(1,
			[] (riscv::Machine<W>& m) {
				try {
					const auto [str] = m.sysargs <std::string> ();
				} catch (...) {}
				try {
					const auto [view] = m.sysargs <std::string_view> ();
				} catch (...) {}
				const auto a0 = m.sysarg(0);
				const auto a1 = m.sysarg(1);
				const auto a2 = m.sysarg(1);

				try {
					std::array<riscv::vBuffer, 128> buffers;
					const auto result =
						m.memory.gather_buffers_from_range(buffers.size(), buffers.data(), a0, a1);
					if (result > buffers.size())
						abort();
				} catch (...) {}
				try {
					const auto buffer = m.memory.rvbuffer(a0, a1);
				} catch (...) {}
				try {
					m.memory.strlen(a0);
				} catch (...) {}
				try {
					if (a2 < 65536)
						m.memory.memcmp(a0, a1, a2);
				} catch (...) {}
				try {
					if (a2 < 65536)
						m.memory.memcpy(a0, m, a1, a2);
				} catch (...) {}
			});

		machine.setup_argv({"program"}, {"LC_ALL=C"});

		machine.cpu.init_execute_area(data, V, len);
		machine.cpu.jump(V);

		machine.simulate(MAX_CYCLES);
	} catch (const std::exception&) {
		//printf(">>> Exception: %s\n", e.what());
	}
	// Accelerate syscall fuzzing
	try {
		machine.system_call(1);
	} catch (...) {
	}
}

extern "C"
void LLVMFuzzerTestOneInput(const uint8_t* data, size_t len)
{
#if defined(FUZZ_ELF)
	fuzz_elf_loader(data, len);
#elif defined(FUZZ_VM)
	fuzz_instruction_set(data, len);
#elif defined(FUZZ_NAT)
	fuzz_native_syscalls(data, len);
#elif defined(FUZZ_SYSH)
	fuzz_syscall_helpers(data, len);
#else
	#error "Unknown fuzzing mode"
#endif
}
