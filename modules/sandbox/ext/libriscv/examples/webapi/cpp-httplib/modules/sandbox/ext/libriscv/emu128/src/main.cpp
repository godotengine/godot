#include <libriscv/debug.hpp>
#include <inttypes.h>
#include <chrono>
static std::string load_file(const std::string&);
using Machine = riscv::Machine<riscv::RISCV128>;
using address_t = riscv::address_type<riscv::RISCV128>;

static void init_program_at(Machine& machine,
	__uint128_t base_addr, const uint8_t* bin, size_t bin_len)
{
	machine.memory.set_page_attr(base_addr, 0xA000, {.read = true, .write = false, .exec = true});
	machine.copy_to_guest(base_addr, bin, bin_len);
	machine.cpu.init_execute_area(bin, base_addr, bin_len);
	machine.cpu.jump(base_addr);
}

int main(int argc, const char** argv)
{
	if (argc < 2) {
		fprintf(stderr, "128-bit ELF required\n");
		exit(1);
	}

	auto binary = load_file(argv[1]);

	Machine machine { binary, {
		.verbose_loader = getenv("VERBOSE") != nullptr
	}};

	/* Install a system call handler that stops the machine. */
	Machine::install_syscall_handler(1,
	 [] (Machine& machine) {
		 machine.stop();
	 });

	 /* Install a system call handler that prints something. */
	Machine::install_syscall_handler(2,
 	 [] (Machine& machine) {
 		 const auto [str] = machine.sysargs <std::string> ();
 		 printf(">>> Program says: %s\n", str.c_str());
 	 });

	/* Add program arguments on the stack. */
	machine.setup_argv({"emu128", "Hello World"});

	auto t0 = std::chrono::high_resolution_clock::now();

	/* This function will run until the exit syscall has stopped the
	   machine, an exception happens which stops execution, or the
	   instruction counter reaches the given limit (1M): */
	try {
		if (getenv("DEBUG") != nullptr) {
			riscv::DebugMachine debugger { machine };
			debugger.verbose_instructions = true;

			debugger.simulate(50'000'000UL);
		} else {
			machine.simulate(15'000'000'000ULL);
		}
	} catch (const std::exception& e) {
		fprintf(stderr ,"%s\n", machine.cpu.current_instruction_to_string().c_str());
		fprintf(stderr, ">>> Runtime exception: %s\n", e.what());
	}

	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> runtime = t1 - t0;

	if (getenv("DEBUG") != nullptr) {
		printf("\n\nFinal machine registers:\n%s\n",
			machine.cpu.registers().to_string().c_str());
	}
	// You can silence this output by setting SILENT=1, like so:
	// SILENT=1 ./rvlinux myprogram
	if (getenv("SILENT") == nullptr) {
		const auto retval = machine.return_value<uint64_t>();

		printf(">>> Program exited, exit code = %" PRId64 " (0x%" PRIX64 ")\n",
			int64_t(retval), uint64_t(retval));
		printf("Instructions executed: %" PRIu64 "  Runtime: %.3fms  Insn/s: %.0fmi/s\n",
			machine.instruction_counter(), runtime.count()*1000.0,
			machine.instruction_counter() / (runtime.count() * 1e6));
		printf("Pages in use: %zu (%zu kB virtual memory, total %zu kB)\n",
			machine.memory.pages_active(),
			machine.memory.pages_active() * 4,
			machine.memory.memory_usage_total() / 1024UL);
	}
}

#include <stdexcept>
#include <unistd.h>
std::string load_file(const std::string& filename)
{
    size_t size = 0;
    FILE* f = fopen(filename.c_str(), "rb");
    if (f == NULL) throw std::runtime_error("Could not open file: " + filename);

    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string result;
	result.resize(size);
    if (size != fread(result.data(), 1, size, f))
    {
        fclose(f);
        throw std::runtime_error("Error when reading from file: " + filename);
    }
    fclose(f);
    return result;
}
