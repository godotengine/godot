#include <libriscv/machine.hpp>
static inline std::vector<uint8_t> load_file(const std::string&);

int main(int argc, const char** argv)
{
	// load binary from file
	const std::vector<uint8_t> binary = load_file(argv[1]);

	using namespace riscv;
	Machine<RISCV64> machine { binary };
	// install a system call handler
	Machine<RISCV64>::install_syscall_handler(93,
	 [] (Machine<RISCV64>& machine) {
		 const auto [code] = machine.sysargs <int> ();
		 printf(">>> Program exited, exit code = %d\n", code);
		 machine.stop();
	 });

	// add program arguments on the stack
	const std::vector<std::string> args = {
		"emulator"
	};
	const std::vector<std::string> env = {
		"LC_CTYPE=C", "LC_ALL=C", "USER=groot"
	};
	machine.setup_linux(args, env);
	// some extra syscalls
	machine.setup_linux_syscalls();
	// multi-threading
	machine.setup_posix_threads();

	// this function will run until the exit syscall has stopped the
	// machine, an exception happens which stops execution, or the
	// instruction counter reaches the given limit (1M):
	try {
		machine.simulate(1'000'000);
	} catch (const std::exception& e) {
		fprintf(stderr, ">>> Runtime exception: %s\n", e.what());
	}
}

#include <stdexcept>
#include <unistd.h>
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
