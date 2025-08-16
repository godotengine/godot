#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <libriscv/machine.hpp>
#include <libriscv/rv32i_instr.hpp>
#include <any>
#include "custom.hpp"
extern std::vector<uint8_t> build_and_load(const std::string& code,
	const std::string& args = "-O2 -static", bool cpp = false);
static const uint64_t MAX_MEMORY = 8ul << 20; /* 8MB */
static const uint64_t MAX_INSTRUCTIONS = 10'000'000ul;
static const std::string cwd {SRCDIR};
using namespace riscv;

struct InstructionState
{
	std::array<std::any, 8> args;
};

/** The new custom instruction **/
static const Instruction<RISCV64> custom_instruction_handler
{
	[] (CPU<RISCV64>& cpu, rv32i_instruction instr) {
		printf("Hello custom instruction World!\n");
		REQUIRE(instr.opcode() == 0b1011011);

		auto* state = cpu.machine().get_userdata<InstructionState> ();
		// Argument number
		const unsigned idx = instr.Itype.rd & 7;
		// Select type and retrieve value from argument registers
		switch (instr.Itype.funct3)
		{
		case 0x0: // Register value (64-bit unsigned)
			state->args[idx] = cpu.reg(REG_ARG0 + idx);
			break;
		case 0x1: // 64-bit floating point
			state->args[idx] = cpu.registers().getfl(REG_FA0 + idx).f64;
			break;
		default:
			throw "Implement me";
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) {
		return snprintf(buffer, len, "CUSTOM: 4-byte 0x%X (0x%X)",
						instr.opcode(), instr.whole);
	}
};

TEST_CASE("Custom instruction", "[Custom]")
{
	// Build a program that uses a custom instruction to
	// select and identify a system call argument.
	const auto binary = build_and_load(R"M(
int main()
{
	__asm__("li t0, 1234");           // Load integer in T0
	__asm__("fcvt.d.w fa1, t0");      // Move integer from T0 to FA1 (64-bit fp)
	__asm__("li a3, 0xDEADB33F");     // Load integer in A3
	__asm__("li a7, 500");            // System call number 500
	__asm__(".word 0b1000011011011"); // Indicate F1 contains a 64-bit fp argument
	__asm__(".word 0b0000111011011"); // Indicate A3 contains a 64-bit unsigned argument
	__asm__("ecall");                 // Execute system call
	__asm__("ret");
}
)M");

	// Install the handler for unimplemented instructions, allowing us to
	// select our custom instruction for a reserved opcode.
	CPU<RISCV64>::on_unimplemented_instruction =
	[] (rv32i_instruction instr) -> const Instruction<RISCV64>& {
		if (instr.opcode() == 0b1011011) {
			return custom_instruction_handler;
		}
		return CPU<RISCV64>::get_unimplemented_instruction();
	};

	// Install system call number 500 (used by our program above).
	static bool syscall_was_called = false;
	Machine<RISCV64>::install_syscall_handler(500,
	[] (Machine<RISCV64>& machine) {
		auto* state = machine.get_userdata<InstructionState> ();

		REQUIRE(std::any_cast<double>(state->args[1]) == 1234.0);
		REQUIRE(std::any_cast<uint64_t>(state->args[3]) == 0xDEADB33F);
		syscall_was_called = true;
	});

	InstructionState state;

	// Normal (fastest) simulation
	{
		riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
		machine.set_userdata(&state);
		// We need to install Linux system calls for maximum gucciness
		machine.setup_linux_syscalls();
		// We need to create a Linux environment for runtimes to work well
		machine.setup_linux(
			{"custom_instruction"},
			{"LC_TYPE=C", "LC_ALL=C", "USER=root"});
		// Run for at most X instructions before giving up
		syscall_was_called = false;
		machine.simulate(MAX_INSTRUCTIONS);
		REQUIRE(syscall_was_called == true);
	}
	// Precise (step-by-step) simulation
	{
		riscv::Machine<RISCV64> machine{binary, { .memory_max = MAX_MEMORY }};
		machine.set_userdata(&state);
		machine.setup_linux_syscalls();
		machine.setup_linux(
			{"custom_instruction"},
			{"LC_TYPE=C", "LC_ALL=C", "USER=root"});
		// Verify step-by-step simulation
		syscall_was_called = false;
		machine.set_max_instructions(MAX_INSTRUCTIONS);
		machine.cpu.simulate_precise();
		REQUIRE(syscall_was_called == true);
	}
}

#include <map>
struct SystemFunctionHandler {
	std::function<SystemArg(Machine<RISCV64>&, const SystemFunctionArgs&)> handler;
	size_t arguments = 0;
};
static std::map<std::string, SystemFunctionHandler> sf_handlers;

static void add_system_functions()
{
	sf_handlers["AddTwoFloats"].handler =
		[] (Machine<RISCV64>&, const SystemFunctionArgs& args) -> SystemArg {
			// TODO: Check arguments
			printf("AddTwoFloats: %f + %f = %f\n",
				args.arg[0].f32, args.arg[1].f32, args.arg[0].f32 + args.arg[1].f32);
			return {
				.f32 = args.arg[0].f32 + args.arg[1].f32,
				.type = FLOAT_32,
			};
		};
	sf_handlers["AddTwoFloats"].arguments = 2;

	sf_handlers["Print"].handler =
		[] (Machine<RISCV64>&, const SystemFunctionArgs& args) -> SystemArg {
			// TODO: Check arguments
			std::string str { args.arg[0].string };
			printf("Print: %s\n", str.c_str());
			REQUIRE(str == "Hello World!");
			return {
				.u32 = (unsigned)str.size(),
				.type = UNSIGNED_INT,
			};
		};
	sf_handlers["Print"].arguments = 1;
}

static SystemArg perform_system_function(Machine<RISCV64>& machine,
	const std::string& name, size_t argc, SystemFunctionArgs& args)
{
	printf("System function: %s\n", name.c_str());

	auto it = sf_handlers.find(name);
	if (it == sf_handlers.end())
	{
		fprintf(stderr, "Error: No such system function: %s\n", name.c_str());
		return {
			.u32 = ERROR_NO_SUCH_FUNCTION,
			.type = ERROR,
		};
	}
	auto& handler = it->second;

	if (argc < handler.arguments)
	{
		fprintf(stderr, "Error: Missing arguments to system function: %s\n", name.c_str());
		return {
			.u32 = ERROR_MISSING_ARGUMENTS,
			.type = ERROR
		};
	}

	// Zero-terminate all strings (set the last char to zero)
	for (size_t i = 0; i < argc; i++) {
		if (args.arg[i].type == STRING)
			args.arg[i].string[STRING_BUFFER_SIZE-1] = 0;
	}

	return handler.handler(machine, args);
}

TEST_CASE("Take custom system arguments", "[Custom]")
{
	const auto binary = build_and_load(R"M(
	#include "custom.hpp"
	#include <stdio.h>
	#include <string.h>

	static void system_function(
		const char *name,
		size_t n, struct SystemFunctionArgs *args,
		struct SystemArg *result)
	{
		register const char                *a0 __asm__("a0") = name;
		register size_t                     a1 __asm__("a1") = n;
		register struct SystemFunctionArgs *a2 __asm__("a2") = args;
		register struct SystemArg          *a3 __asm__("a3") = result;
		register long               syscall_id __asm__("a7") = 500;
		register long                   a0_out __asm__("a0");

		__asm__ volatile ("scall"
			: "=r"(a0_out), "+m"(*a3)
			: "r"(a0), "m"(*a0), "r"(a1), "r"(a2), "m"(*a2), "r"(a3), "r"(syscall_id));
		(void)a0_out;
	}

	static void print_arg(struct SystemArg *arg)
	{
		switch (arg->type) {
			case SIGNED_INT:
				printf("32-bit signed integer: %d\n", arg->i32);
				break;
			case UNSIGNED_INT:
				printf("32-bit unsigned integer: %d\n", arg->u32);
				break;
			case FLOAT_32:
				printf("32-bit floating-point: %f\n", arg->f32);
				break;
			case FLOAT_64:
				printf("64-bit floating-point: %f\n", arg->f64);
				break;
			case STRING:
				printf("String: %s\n", arg->string);
				break;
			case ERROR:
				printf("Error code: 0x%X\n", arg->u32);
				break;
			default:
				printf("Unknown value: 0x%X\n", arg->u32);
		}
	}

	int main() {
		// Setup system function "AddTwoFloats"
		struct SystemFunctionArgs sfa;
		sfa.arg[0].type = FLOAT_32;
		sfa.arg[0].f32  = 64.0f;
		sfa.arg[1].type = FLOAT_32;
		sfa.arg[1].f32  = 32.0f;

		// Perform 'AddTwoFloats' system function
		struct SystemArg result;
		system_function("AddTwoFloats", 2, &sfa, &result);

		// Result should be a 32-bit FP value
		print_arg(&result);

		// Perform 'Print'
		sfa.arg[0].type = STRING;
		strcpy(sfa.arg[0].string, "Hello World!");
		system_function("Print", 1, &sfa, &result);

		return 0x1234;
	})M", "-O2 -static -I" + cwd);

	Machine<RISCV64> machine{binary};
	machine.setup_linux(
		{"myprogram"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});
	machine.setup_linux_syscalls();

	// Add our system functions
	add_system_functions();

	Machine<RISCV64>::install_syscall_handler(500,
	[] (Machine<RISCV64>& machine) {
		// Retrieve name (string), argument count (32-bit unsigned)
		// and the whole SystemFunctionArgs structure.
		auto [name, argc, args] =
			machine.sysargs <std::string, unsigned, SystemFunctionArgs> ();
		// The address of the result
		auto g_result = machine.sysarg(3);
		// A little bounds-checking
		const size_t count = std::min(argc, 4u);
		auto result =
			perform_system_function(machine, name, count, args);
		machine.copy_to_guest(g_result, &result, sizeof(result));
		machine.set_result(0);
	});

	static bool found = false;
	machine.set_printer([] (const auto&, const char* data, size_t size) {
		std::string text{data, data + size};
		if (text == "32-bit floating-point: 96.000000" // musl
			|| text == "32-bit floating-point: 96.000000\n") // glibc
			found = true;
	});

	machine.simulate();

	REQUIRE(machine.return_value() == 0x1234);
	REQUIRE(found == true);
}
