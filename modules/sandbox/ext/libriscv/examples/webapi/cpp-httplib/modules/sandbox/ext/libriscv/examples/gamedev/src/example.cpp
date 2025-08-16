#include "event.hpp"
#include <chrono>
#include <fmt/core.h>
#include <libriscv/rsp_server.hpp>
#include <libriscv/rv32i_instr.hpp>
using namespace riscv;
template <unsigned SAMPLES = 2000>
static void benchmark(std::string_view name, Script& script, std::function<void()> fn);

// ScriptCallable is a function that can be requested from the script
using ScriptCallable = std::function<void(Script&)>;
// A map of host functions that can be called from the script
static std::array<ScriptCallable, 64> g_script_functions {};
static void register_script_function(uint32_t number, ScriptCallable&& fn) {
	g_script_functions.at(number) = std::move(fn);
}

void Script::setup_syscall_interface()
{
	// ALTERNATIVE 1:
	// A custom system call that executes a function based on an index
	// The most common approach, but requires registers T0 and A7 to be used
	Script::machine_t::install_syscall_handler(510,
	[] (Script::machine_t& machine) {
		Script& script = *machine.template get_userdata<Script>();
		g_script_functions.at(machine.cpu.reg(riscv::REG_T0))(script);
	});

	// ALTERNATIVE 2:
	// A custom instruction that executes a function based on an index
	// This variant is faster than a system call, and can use 8 integers as arguments
    using namespace riscv;
    static const Instruction<MARCH> unchecked_dyncall_instruction {
        [](CPU<MARCH>& cpu, riscv::rv32i_instruction instr)
        {
            Script& script = *cpu.machine().template get_userdata<Script>();
            g_script_functions[instr.Itype.imm](script);
        },
        [](char* buffer, size_t len, auto&, riscv::rv32i_instruction instr) -> int
        {
            return fmt::format_to_n(buffer, len,
                "DYNCALL: 4-byte idx={:x} (inline, 0x{:X})",
                uint32_t(instr.Itype.imm),
                instr.whole
            ).size;
        }};
    // Override the machines unimplemented instruction handling,
    // in order to use the custom instruction for a given opcode.
    CPU<MARCH>::on_unimplemented_instruction
        = [](riscv::rv32i_instruction instr) -> const Instruction<MARCH>&
    {
        if (instr.opcode() == 0b1011011 && instr.Itype.rs1 == 0 && instr.Itype.rd == 0)
        {
			if (instr.Itype.imm < g_script_functions.size())
				return unchecked_dyncall_instruction;
        }
        return CPU<MARCH>::get_unimplemented_instruction();
    };
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		fmt::print("Usage: {} [program file] [arguments ...]\n", argv[0]);
		return -1;
	}

	// Register a custom function that can be called from the script
	// This is the handler for dyncall1
	register_script_function(1, [](Script& script) {
		auto [arg] = script.machine().sysargs<int>();

		fmt::print("dyncall1 called with argument: 0x{:x}\n", arg);

		script.machine().set_result(42);
	});
	// This is the handler for dyncall2
	register_script_function(2, [](Script& script) {
		// string_view consumes 2 arguments, the first is the pointer, the second is the length
		// unlike std::string, which consumes only 1 argument (zero-terminated string pointer)
		auto [view, str] = script.machine().sysargs<std::string_view, std::string>();

		fmt::print("dyncall2 called with arguments: '{}' and '{}'\n", view, str);
	});
	// This is the handler for dyncall_empty
	register_script_function(3, [](Script&) {
	});
	// This is the handler for dyncall_data
	register_script_function(4, [](Script& script) {
		struct MyData {
			char buffer[32];
		};
		auto [data_span, data] = script.machine().sysargs<std::span<MyData>, const MyData*>();

		fmt::print("dyncall_data called with args: '{}' and '{}'\n", data_span[0].buffer, data->buffer);
	});
	// This is the handler for dyncall_string
	register_script_function(5, [](Script& script) {
		auto [str] = script.machine().sysargs<GuestStdString<Script::MARCH>*>();

		fmt::print("dyncall_string called: {}\n", str->to_view(script.machine()));
	});

	// Create a new script instance, loading and initializing the given program file
	// The programs main() function will be called
	Script script("myscript", argv[1]);

	// Create an event for the 'test1' function with 4 arguments and returns an int
	Event<int(int, int, int, int)> test1(script, "test1");
	if (auto ret = test1(1, 2, 3, 4))
		fmt::print("test1 returned: {}\n", *ret);
	else
		throw std::runtime_error("Failed to call test1!?");

	// Benchmark the test2 function, which allocates and frees 1024 bytes
	Event<void()> test2(script, "test2");
	if (auto ret = test2(); !ret)
		throw std::runtime_error("Failed to call test2!?");

	benchmark("std::make_unique[1024] alloc+free", script, [&] {
		test2();
	});

	// Create an event for the 'test3' function with a single string argument
	// This function will throw an exception, which is immediately caught
	Event<void(std::string)> test3(script, "test3");
	if (auto ret = test3("Oh, no! An exception!"); !ret)
		throw std::runtime_error("Failed to call test3!?");

	// Pass data structure to the script
	struct Data {
		int a, b, c, d;
		float e, f, g, h;
		double i, j, k, l;
		char buffer[32];
	};
	const Data data = { 1, 2, 3, 4, 5.0f, 6.0f, 7.0f, 8.0f, 9.0, 10.0, 11.0, 12.0, "Hello, World!" };
	Event<void(Data)> test4(script, "test4");
	if (auto ret = test4(data); !ret)
		throw std::runtime_error("Failed to call test4!?");

	// Benchmark the overhead of dynamic calls
	Event<void()> bench_dyncall_overhead(script, "bench_dyncall_overhead");
	benchmark("Overhead of dynamic calls", script, [&] {
		bench_dyncall_overhead();
	});

	// Call test5 function
	Event<void()> test5(script, "test5");
	if (auto ret = test5(); !ret)
		throw std::runtime_error("Failed to call test5!?");

	// For C++ programs we have some guest abstractions we can test
	// like std::string and std::vector wrappers.
	if (script.address_of("test6"))
	{
		// A scoped guest object is a C++ object that is allocated and freed on the guest heap,
		// with lifetime tied to the current scope.
		using CppString       = ScopedGuestStdString<Script::MARCH>;
		using CppVectorInt    = ScopedGuestStdVector<Script::MARCH, int>;
		using CppVectorString = ScopedGuestStdVector<Script::MARCH, GuestStdString<Script::MARCH>>;

		// Define the test6 function, which has a std::string& argument in the guest,
		// and a std::vector<int>& and std::vector<std::string>&. The stack is writable,
		// so the guest can choose whether or not to use const references.
		Event<void(CppString&, CppVectorInt&, CppVectorString&)> test6(script, "test6");

		for (int i = 0; i < 1; i++) {
			// Create a GuestStdString on the guest heap
			CppString str(script.machine(), "C++ World ..SSO..");
			// Create a GuestStdVector of ints on the guest heap
			CppVectorInt ivec(script.machine(), std::vector<int>{ 1, 2, 3, 4, 5 });
			// Create a GuestStdVector of strings on the guest heap
			CppVectorString svec(script.machine(),
				std::vector<std::string>{ "Hello,", "World!", "This string is long :)" });

			// Call the test6 function with all the objects
			if (auto ret = test6(str, ivec, svec); !ret)
				throw std::runtime_error("Failed to call test6!?");
		}
	}

	// If GDB=1, start the RSP server for debugging
	if (getenv("GDB"))
	{
		// Setup the test6 function to be debugged from GDB
		script.machine().setup_call();
		script.machine().cpu.jump(script.machine().address_of("remote_debug_test"));
		// Start the RSP server on port 2159
		fmt::print("Waiting for GDB to connect on port 2159...\n");
		riscv::RSP rsp(script.machine(), 2159);
		if (auto client = rsp.accept(); client) {
			fmt::print("GDB connected\n");
			while (client->process_one());
			fmt::print("GDB session ended\n");
		} else {
			fmt::print("Failed to accept GDB connection. Waited too long?\n");
		}
	}

}

// A simple benchmarking function that subtracts the call overhead
template <unsigned SAMPLES>
void benchmark(std::string_view name, Script &script, std::function<void()> fn)
{
	static unsigned overhead = 0;
	if (overhead == 0)
	{
		Event<void()> measure_overhead(script, "measure_overhead");
		auto start = std::chrono::high_resolution_clock::now();
		for (unsigned i = 0; i < SAMPLES; i++)
			measure_overhead();
		auto end = std::chrono::high_resolution_clock::now();
		overhead = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / SAMPLES;
		fmt::print("Call overhead: {}ns\n", overhead);
	}

	auto start = std::chrono::high_resolution_clock::now();
	for (unsigned i = 0; i < SAMPLES; i++)
		fn();
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / SAMPLES;
	fmt::print("Benchmark: {}  Elapsed time: {}ns\n",
			   name, elapsed - overhead);
}
