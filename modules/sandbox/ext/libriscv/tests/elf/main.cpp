#include <libriscv/debug.hpp>
#include <inttypes.h>
#include <stdexcept>
#include <unistd.h>
static inline std::vector<uint8_t> load_file(const std::string&);
static constexpr uint64_t MAX_MEMORY = 1024 * 1024 * 256;

template <int W>
static long run_program(
	const char* program,
	const std::vector<uint8_t>& binary,
	const std::vector<std::string>& args)
{
	struct State {
		const char* program;
	} state;
	state.program = program;

	riscv::Machine<W> machine { binary, {
		.memory_max = MAX_MEMORY,
		.allow_write_exec_segment = true,
		.verbose_loader = (getenv("VERBOSE") != nullptr),
		.use_memory_arena = false,
	}};
	machine.set_userdata(&state);

	machine.on_unhandled_syscall =
	[] (auto& machine, size_t) -> void {
		auto* state = machine.template get_userdata<State> ();
		const auto a0 = machine.cpu.reg(riscv::REG_ARG0);
		const auto gp = machine.cpu.reg(riscv::REG_GP);
		if (a0 == 1) {
			printf("[\033[0;32mPASS\033[0m] %s\n", state->program);
			exit(0);
		}
		printf("[\033[0;31mFAIL\033[0m] GP=%" PRIu64 " %s\n",
			uint64_t(gp), state->program);
		riscv::DebugMachine debug { machine };
		debug.print_and_pause();
		exit(0);
	};

	// A CLI debugger
	riscv::DebugMachine debug { machine };
	//debug.verbose_instructions = true;
	//debug.verbose_fp_registers = true;
	//debug.verbose_registers = true;

	auto addr = machine.address_of("userstart");
	if (addr) machine.cpu.jump(addr);

	try {
		debug.simulate();
	} catch (riscv::MachineException& me) {
		printf("%s\n", machine.cpu.current_instruction_to_string().c_str());
		printf(">>> Machine exception %d: %s (data: 0x%" PRIX64 ")\n",
				me.type(), me.what(), me.data());
		printf("%s\n", machine.cpu.registers().to_string().c_str());
		machine.memory.print_backtrace(
			[] (std::string_view line) {
				printf("-> %.*s\n", (int)line.size(), line.begin());
			});
		if (me.type() == riscv::UNIMPLEMENTED_INSTRUCTION || me.type() == riscv::MISALIGNED_INSTRUCTION) {
			printf(">>> Is an instruction extension disabled?\n");
			printf(">>> A-extension: %d  C-extension: %d  V-extension: %d\n",
				riscv::atomics_enabled, riscv::compressed_enabled, riscv::vector_extension);
		}
		debug.print_and_pause();
	} catch (std::exception& e) {
		printf(">>> Exception: %s\n", e.what());
		machine.memory.print_backtrace(
			[] (std::string_view line) {
				printf("-> %.*s\n", (int)line.size(), line.begin());
			});
		debug.print_and_pause();
	}

	throw std::runtime_error("Execution ended without system call");
}

int main(int argc, const char** argv)
{
	if (argc < 2) {
		fprintf(stderr, "Provide RISC-V binary as argument!\n");
		return 1;
	}

	std::vector<std::string> args;
	for (int i = 1; i < argc; i++) {
		args.push_back(argv[i]);
	}
	const std::string& filename = args.front();

	const auto binary = load_file(filename);
	assert(binary.size() >= 64);

	try {
		if (binary[4] == ELFCLASS64)
			return run_program<riscv::RISCV64> (filename.c_str(), binary, args);
		else
			return run_program<riscv::RISCV32> (filename.c_str(), binary, args);
	} catch (const std::exception& e) {
		fprintf(stderr,
			"Exception: %s\n", e.what());
	}

	printf("[FAIL] %s\n", filename.c_str());
	return 1;
}

std::vector<uint8_t> load_file(const std::string& filename)
{
    size_t size = 0;
    FILE* f = fopen(filename.c_str(), "rb");
    if (f == NULL) throw std::runtime_error("Could not open file: " + filename);

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
