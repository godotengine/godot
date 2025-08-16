#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <libriscv/machine.hpp>
extern std::vector<uint8_t> build_and_load(const std::string& code,
	const std::string& args = "-O2 -static", bool cpp = false);
using namespace riscv;

TEST_CASE("Main example", "[Examples]")
{
	const auto binary = build_and_load(R"M(
	extern void exit(int);
	int main() {
		exit(666);
		return 123;
	})M");

	Machine<RISCV64> machine { binary };
	machine.setup_linux(
		{"myprogram", "1st argument!", "2nd argument!"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});
	machine.setup_linux_syscalls();

	struct State {
		long code = -1;
	} state;
	machine.set_userdata(&state);

	// exit and exit_group
	Machine<RISCV64>::install_syscall_handler(94,
		[] (Machine<RISCV64>& machine) {
			const auto [code] = machine.sysargs <int> ();

			auto& state = *machine.get_userdata<State> ();
			state.code = code;

			machine.stop();
		});
	// Newlib uses regular exit syscall (93)
	Machine<RISCV64>::install_syscall_handler(93,
		Machine<RISCV64>::syscall_handlers.at(94));

	machine.simulate(5'000'000UL);

	REQUIRE(state.code == 666);
	REQUIRE(machine.return_value() == 666);
}

#include <libriscv/rv32i_instr.hpp>

TEST_CASE("One instruction at a time", "[Examples]")
{
	const auto binary = build_and_load(R"M(
	extern void exit(int);
	int main() {
		return 0x1234;
	})M");

	Machine<RISCV64> machine{binary};
	machine.setup_linux(
		{"myprogram"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});
	machine.setup_linux_syscalls();

	machine.set_max_instructions(1'000'000UL);

	while (!machine.stopped()) {
		auto& cpu = machine.cpu;
		// Read next instruction
		auto instruction = cpu.read_next_instruction();
		// Print the instruction to terminal
		printf("%s\n",
			   cpu.to_string(instruction, cpu.decode(instruction)).c_str());
		// Execute instruction directly
		cpu.execute(instruction);
		// Increment PC to next instruction, and increment instruction counter
		cpu.increment_pc(instruction.length());
		machine.increment_counter(1);
	}

	REQUIRE(machine.return_value() == 0x1234);
}

TEST_CASE("One instruction at a time with ilimit", "[Examples]")
{
	const auto binary = build_and_load(R"M(
	int main() {
		return 0x1234;
	})M");

	Machine<RISCV64> machine{binary};
	machine.setup_linux(
		{"myprogram"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});
	machine.setup_linux_syscalls();

	do {
		// Only execute 1000 instructions at a time
		machine.reset_instruction_counter();
		machine.set_max_instructions(1'000);

		while (!machine.stopped())
		{
			auto& cpu = machine.cpu;
			// Read next instruction
			const auto instruction = cpu.read_next_instruction();
			// Print the instruction to terminal
			printf("%s\n", cpu.to_string(instruction).c_str());
			// Execute instruction directly
			cpu.execute(instruction);
			// Increment PC to next instruction, and increment instruction counter
			cpu.increment_pc(instruction.length());
			machine.increment_counter(1);
		}

	} while (machine.instruction_limit_reached());

	REQUIRE(machine.return_value() == 0x1234);
}

TEST_CASE("Build machine from empty", "[Examples]")
{
	Machine<RISCV32> machine;
	machine.setup_minimal_syscalls();

	std::vector<uint32_t> my_program {
		0x29a00513, //        li      a0,666
		0x05d00893, //        li      a7,93
		0x00000073, //        ecall
	};

	// Set main execute segment (12 instruction bytes)
	const uint32_t dst = 0x1000;
	machine.cpu.init_execute_area(my_program.data(), dst, 12);

	// Jump to the start instruction
	machine.cpu.jump(dst);

	// Geronimo!
	machine.simulate(1'000ul);

	REQUIRE(machine.return_value() == 666);
}

TEST_CASE("Execute while doing other things", "[Examples]")
{
	const auto binary = build_and_load(R"M(
	__attribute__((used, retain))
	long test() {
		for (volatile unsigned i = 0; i < 10000; i++);
		return 0x5678;
	}
	int main() {
		return 0x1234;
	})M");

	Machine<RISCV64> machine{binary};
	machine.setup_linux(
		{"myprogram"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});
	machine.setup_linux_syscalls();

	machine.simulate();
	REQUIRE(machine.return_value() == 0x1234);

	auto test_addr = machine.address_of("test");

	// Reset the stack pointer from any previous call to its initial value
	machine.cpu.reset_stack_pointer();
	// Function call setup for the guest VM, but don't start execution
	machine.setup_call(555, 666);
	machine.cpu.jump(test_addr);
	// Run the program for X amount of instructions, then print something, then
	// resume execution again. Do this until stopped.
	do {
		// Execute 1000 instructions at a time
		machine.simulate<false>(1000);
		// Do some work in between simulation
		printf("Working ...\n");
	} while (machine.instruction_limit_reached());

	REQUIRE(machine.return_value() == 0x5678);
}
